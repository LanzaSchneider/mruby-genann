mruby-genann
=================================================

Genann is a light-weight ANN library
-

This is a mruby binding.(Genann has been bundled in.)
-

Usage(also can found in ```./test/genann.rb```)
--

```Ruby
# Input and expected out data for the XOR function.
input = [[0, 0], [0, 1], [1, 0], [1, 1]]
output = [[0], [1], [1], [0]]
# Use Array#pack for dataset passing.
input_buf = input.collect {|arr| arr.pack('E*') }
output_buf = output.collect {|arr| arr.pack('E*') }
# New network with 2 inputs,
# 1 hidden layer of 2 neurons,
# and 1 output.
ann = Genann.new(2, 1, 2, 1)
# randomize weights.
srand
ann.weights_load Array.new(ann.total_weights){ rand }.pack('E*')
# Train on the four labeled data points many times.
500.times do
 ann.train input_buf[0], output_buf[0], 3
 ann.train input_buf[1], output_buf[1], 3
 ann.train input_buf[2], output_buf[2], 3
 ann.train input_buf[3], output_buf[3], 3
end
# Run the network and see what it predicts.
# Use String#unpack for result parsing.
puts sprintf('Output for [%1.f, %1.f] is %1.f.', input[0][0], input[0][1], ann.run(input_buf[0]).unpack('E*')[0])
puts sprintf('Output for [%1.f, %1.f] is %1.f.', input[1][0], input[1][1], ann.run(input_buf[1]).unpack('E*')[0])
puts sprintf('Output for [%1.f, %1.f] is %1.f.', input[2][0], input[2][1], ann.run(input_buf[2]).unpack('E*')[0])
puts sprintf('Output for [%1.f, %1.f] is %1.f.', input[3][0], input[3][1], ann.run(input_buf[3]).unpack('E*')[0])
```
