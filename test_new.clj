(ns iris-neutral.test-new)
(require 'clojure.string)
(require 'clojure.data)
(use 'clojure.java.io)
;15.33 is the average over 100 runs
;IndexOutOfBoundsException   clojure.lang.RT.nthFrom (RT.java:795) for hiddens <= 2
;NEURON DEFINITIONS AND CALCULATIONS
;LOOKS LIKE 7 IS THE BEST NEURON COUNT FOR NOW W/ RATE OF .4
(defn print-ret [x] (println x) x)
(def hidden-count 5)
(def output-count 3)
(def rate 0.5)
(def calculate #{4, 5, 6})
(def input-count 4)
(def raw-input ;remove them from input if our calculate contains their index
                             (map read-string (with-open [rdr (reader "new-iris.txt")] ;read the first line as a list of inputs
                                  (clojure.string/split (first (line-seq rdr)) #","))))
;Creates an array of length num inputs of values from -1 (incluse) to 1 (exclusive) and then adds a +1 bias
(defn make-neuron [to_input] 
                (concat (into-array (for [x (range to_input)] 
                                      (- (rand 2) 1))) [1]))
(defn corrector [net] (keep-indexed #(if (contains? calculate %1) %2 nil) (first net)));(drop 1 net)
(defn inputer [net] (keep-indexed #(if (contains? calculate %1) nil %2) (first net)));(drop 1 net)

;(def hidden (for [x (range hidden-count)]  (make-neuron input-count))) ;The hidden layer!
;(def outers (for [x (range output-count)]  (make-neuron (count hidden)))) ; the output layer! THIS COULD BE DONE DIFFERENTLY WITH US DIRECTLY MAKING IT BELOW BUT I DO NOT CARE RIGHT NOW. This the only place this is being used
;(def network (concat (list raw-input) (list hidden) (list outers))) ;A list of layers, with each layer being a list of neurons, and with each neuron being a list of weightss

;(defn logistic [x] (/ 1 (+ 1 (Math/exp (- 0 x)))))
;(defn result-neuron [inputs, weights]  (logistic ( + (reduce + (map * inputs weights)) (last weights))))

(defn result-neuron [inputs, weights]  (Math/tanh ( + (reduce + (map * inputs weights)) (last weights)))) ;(summation of inputs*weights) + bias
(defn result-layer [inputs, layer] (map (partial result-neuron inputs) layer)) ;list of all outputs from the layer with a given input
(defn new-out [net] (drop 0 (reverse (loop [idx 0, seq (list (inputer net))] (if (nth (drop 1 net) idx false) ;0 - input, 1 - hidden, 2 - output
                                                            (recur (+ idx 1) 
                                                                   (conj seq 
                                                                         (result-layer (nth seq 0) 
                                                                                       (nth (drop 1 net) idx)))) 
                                                            seq)))))
;CALCULATE THE ERROR
(defn deriv1 "Eneuron/out" [target, result] (* -1 (- target result)))
(defn deriv2 "out/net" [result] (/ 1 (Math/pow (Math/cosh result) 2))) ;Result is the result of a particular neurons, weights*inputs
;(defn deriv2 "out/net" [result] (* (logistic result) (- 1 (logistic result))))
(defn deriv3 "net/weight" [input] input) ;Input from a single neuron
(defn node-delta [target, result] (* (deriv1 target result) (deriv2 result)))

(defn error-output [target, result, input, net] (* (node-delta target, result)
                                                   (deriv3 input))) ;input from a particular neuron a layer below us

(defn error-hidden [index, result, input, net] (* (reduce + (map * (map node-delta (corrector net) (nth (new-out net) 2)) ;The sum of each node delta on the output layer * the weight from us to them
                                                                 (map #(nth %1 index) (nth net 1)))) ;Right now the layer we are looking forward to is hardcoded, at one point this could be given as an argument once I figure out the math for multilayer correct
                                                  (deriv2 result) 
                                                  (deriv3 input)))
(defn correction-hidden [result, net] (map 
                                        #(concat (map-indexed (fn [indx, input] (* rate (error-hidden indx %1 input net)))
                                                      (nth (new-out net) 0))
                                                 '(0)) ;The '(0) is so that we do not adjust the bias in each neuron
                                        result))
(defn correction-output [result, net]  (map ;We do not need to worry about the bias because we do not change it.
                                         #(concat (map 
                                            (fn [input] (* rate (error-output %1 %2 input, net))) (nth (new-out net) 1))
                                                  '(0))
                                         (corrector net) result)) ;This relies on def, but is good enough for now
(defn correction [index, result, net] (if (= index 0) (correction-hidden result net) (correction-output result net))) ;

;(reduce + (map #(Math/abs %) (map - (last (net-out network)) correct))) ;Would be calculated (scientific) error, but cannot divide by zero.

(defn finale [net] (map (fn [o1, o2] (map #(map - %1 %2) o1 o2)) ;Subtract the innermost value (3rd layer) of the 2nd map from the 1st
                        (drop 1 net) ;The weights
                        (map-indexed (fn [idx, result] (correction idx result net)) (drop 1 (new-out net))))) ;The values to correct (already multiplied by learning rate)
;We need to do some code-review to make sure what is being tested and mapped against is the correct layer
;AFTER THE CODE REVIEW!!! Make the above into a loop recur with a let such that the input is updated
(def big-ol-input (map vec (map (fn [o1] (map read-string o1)) (map #(clojure.string/split % #",") (clojure.string/split-lines (slurp "new-iris.txt"))))))
(defn net-correct [net] (if (= (.indexOf (last (new-out net)) ;Index of the greatest value in (last (new-out net))
                                 (apply max (last (new-out net)))) ;Which is actually the index of highest output neuron
                               (.indexOf (corrector net) 1)) ;Compare if that is the same as the index of the 1 in the input layer 
                          1 0)) ;Return 1 if true or 0 if false
(defn network-run [hiddens, loops] (loop [net (concat (list raw-input)
                                                  (list (for [x (range hiddens)](make-neuron input-count)))
                                                  (list (for [x (range output-count)]  (make-neuron hiddens)))),
                                      count 0, correct 0, runs 0]; (println count runs)
                                               (if (< count 150) 
                                                 (if (< count 120) ;(println "RUN" count net)
                                                   (do  (recur (concat (list (nth big-ol-input count)) ;Finale only returns the correct neurons
                                                                         (finale net)) ;So we must concat the inputs to return to our original layer scheme 
                                                                   (+ count 1)
                                                                   0
                                                                   runs))
                                                   (if (< runs loops)
                                                     (recur (concat (list (nth big-ol-input count))(finale net))
                                                          0
                                                          0
                                                          (+ runs 1))
                                                     (recur (concat (list (nth big-ol-input count))(finale net))
                                                          (+ count 1)
                                                          (+ correct (net-correct net))
                                                          runs)))
                                                      correct)))
;(defn old-run [hiddens] (loop [net (concat (list raw-input)
;                                           (list (for [x (range hiddens)](make-neuron input-count)))
;                                           (list (for [x (range output-count)]  (make-neuron hiddens)))),
;                               count 0, correct 0]
;                          (if (< count 150) (recur (concat (list (nth big-ol-input count))(finale net))
;                                                   (+ count 1)
;                                                   (if (> count 120) (+ correct (net-correct net)) 0)) 
;                                                  correct)))
(defn loop-run [runs, hiddens, loops] (loop [count 0, sum 0] (if (< count runs) (recur (+ count 1)
                                                                                          (+ sum (network-run hiddens loops)))
                                                                  (float (/ sum count)))))
;(for [x (range 10 15)] (println x (loop-run 50 x)))
;(for [x (range 0 1 0.1)] (do (def rate x)
;                           (println x (loop-run 50 7))
;                                      ))
;TAKE A LOOK AT WHY hiddens < 3 DOES NOT WORK
;THIS PROBLEM IS DEEP AND SHOULD NOT BE HAPPENING. IT <b> REALLY </b> NEEDS TO BE FIXED.
;THE ERROR IS IN CORRECTION-HIDDEN
;(println "RUN")
(def donger (concat (list raw-input)
                                                                                (list (for [x (range 2)](make-neuron input-count)))
                                                                                (list (for [x (range output-count)]  (make-neuron 2)))))
;(print donger)
;(println "SPACE")
;(finale donger)
(println "Start");
(finale donger)
;(loop-run 150 6 0)
;THE ERROR IS UNDER ERROR HIDDEN ON THE SECOND LINE
;(network-run 2 0) ;You should start deconstructing each layer by using printret in finale
