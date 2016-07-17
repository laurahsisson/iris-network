(ns iris-neutral.test-new)
(require 'clojure.string)
(require 'clojure.data)
(use 'clojure.java.io)
(defn print-ret "Print and return, only takes on argument" [x] (println x) x) 
(def rate "Learning rate" 0.5)
(def calculate "Which columns to calculate in the dataset" #{4, 5, 6})
(def input-count "How many inputs we have" 4)
(def raw-input ;remove them from input if our calculate contains their index
                             (map read-string (with-open [rdr (reader "new-iris.txt")] ;read the first line as a list of inputs
                                  (clojure.string/split (first (line-seq rdr)) #","))))
;Creates an array of length num inputs of values from -1 (incluse) to 1 (exclusive) and then adds a +1 bias
(defn make-neuron "Makes a neuron with to_input random weights and a bias" [to_input] 
                (concat (into-array (for [x (range to_input)] 
                                      (- (rand 2) 1))) [1]))
(defn corrector "Takes a network and returns the correct values that network should calculate" [net] (keep-indexed #(if (contains? calculate %1) %2 nil) (first net)));(drop 1 net)
(defn inputer "Takes a network and returns the input to the network" [net] (keep-indexed #(if (contains? calculate %1) nil %2) (first net)));(drop 1 net)
(defn result-neuron "Given an input and a neuron, returns the result" [inputs, weights]  (Math/tanh ( + (reduce + (map * inputs weights)) (last weights)))) ;(summation of inputs*weights) + bias
(defn result-layer "Given a layer and the inputs to that layer, returns the result" [inputs, layer] (map (partial result-neuron inputs) layer)) ;list of all outputs from the layer with a given input
(defn new-out [net] "Given a network, returns the output from the output layer" (drop 0 (reverse (loop [idx 0, seq (list (inputer net))] (if (nth (drop 1 net) idx false) ;0 - input, 1 - hidden, 2 - output
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

(defn error-output "Error in the output layer" [target, result, input, net] (* (node-delta target, result)
                                                    (deriv3 input))) ;input from a particular neuron a layer below us

(defn error-hidden "Error in the hidden layer" [index, result, input, net] (* (reduce + (map * (map node-delta (corrector net) (nth (new-out net) 2)) ;The sum of each node delta on the output layer * the weight from us to them
                                                                 (map #(nth %1 index) (nth net 1)))) ;Right now the layer we are looking forward to is hardcoded, at one point this could be given as an argument once I figure out the math for multilayer correct
                                                  (deriv2 result) 
                                                  (deriv3 input)))
(defn correction-hidden "Returns the corrected hidden layer" [result, net] (map 
                                        #(concat (map-indexed (fn [indx, input] (* rate (error-hidden indx %1 input net)))
                                                      (nth (new-out net) 0))
                                                 '(0)) ;The '(0) is so that we do not adjust the bias in each neuron
                                        result))
(defn correction-output "Returns the corrected output layer" [result, net]  (map ;We do not need to worry about the bias because we do not change it.
                                          #(concat (map 
                                             (fn [input] (* rate (error-output %1 %2 input, net))) (nth (new-out net) 1))
                                                   '(0))
                                          (corrector net) result)) ;This relies on def, but is good enough for now
(defn correction "Returns a corrected layer, like the above two, but works with either output or hidden" [index, result, net] (if (= index 0) (correction-hidden result net) (correction-output result net)))

(defn finale [net] "Returns the corrected network" (map (fn [o1, o2] (map #(map - %1 %2) o1 o2)) ;Subtract the innermost value (3rd layer) of the 2nd map from the 1st
                       (drop 1 net) ;The weights
                       (map-indexed (fn [idx, result] (correction idx result net)) (drop 1 (new-out net))))) ;The values to correct (already multiplied by learning rate)

(def big-ol-input "The entirety of the input file, as a vector" (map vec (map (fn [o1] (map read-string o1)) (map #(clojure.string/split % #",") (clojure.string/split-lines (slurp "new-iris.txt"))))))
(defn net-correct  "Returns 1 if the network outputted the correct result with the input" [net] (if (= (.indexOf (last (new-out net)) ;Index of the greatest value in (last (new-out net))
                               (apply max (last (new-out net)))) ;Which is actually the index of highest output neuron
                             (.indexOf (corrector net) 1)) ;Compare if that is the same as the index of the 1 in the input layer 
                        1 0)) ;Return 1 if true or 0 if false
(defn network-run "Run through the input loops times with hiddens neuron in the hidden layer, returns the number of correct outputs" [hiddens, loops] (loop [net (concat (list raw-input)
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
(defn loop-run "As above, but goes through runs times and averages the result"[runs, hiddens, loops] (loop [count 0, sum 0] (if (< count runs) (recur (+ count 1)
                                                                                          (+ sum (network-run hiddens loops)))
                                                                  (float (/ sum count)))))
(loop-run 50 15 0)
