mruby-genann
=================================================
Genann is a light-weight ANN library.
-
This is a mruby binding.(Genann has been bundled in.)
-

Usage:
--

```Ruby
# XOR TEST

# /* Creates and returns a new ann. */
# genann *genann_init(int32_t inputs, int32_t hidden_layers, int32_t hidden, int32_t outputs);
genann = Genann.new(2, 1, 2, 1)

inputs = [
	Genann::Array.new([0, 0]),
	Genann::Array.new([0, 1]),
	Genann::Array.new([1, 0]),
	Genann::Array.new([1, 1])
];

outputs = Genann::Array.new([0, 1, 1, 0])

300.times do
	# Genann train:
	# args: inputs, outputs, outputs_index, learning_rate
	genann.train inputs[0], outputs, 0, 3
	genann.train inputs[1], outputs, 1, 3
	genann.train inputs[2], outputs, 2, 3
	genann.train inputs[3], outputs, 3, 3
end

print sprintf("Output for [%1.f, %1.f] is %1.f.\n", inputs[0][0], inputs[0][1], genann.run(inputs[0]))
print sprintf("Output for [%1.f, %1.f] is %1.f.\n", inputs[1][0], inputs[1][1], genann.run(inputs[1]))
print sprintf("Output for [%1.f, %1.f] is %1.f.\n", inputs[2][0], inputs[2][1], genann.run(inputs[2]))
print sprintf("Output for [%1.f, %1.f] is %1.f.\n", inputs[3][0], inputs[3][1], genann.run(inputs[3]))
```

Serialization:
--

```Ruby
# XOR TEST
genann = Genann.new(2, 1, 2, 1)

inputs = [
	Genann::Array.new([0, 0]),
	Genann::Array.new([0, 1]),
	Genann::Array.new([1, 0]),
	Genann::Array.new([1, 1])
];

outputs = Genann::Array.new([0, 1, 1, 0])

300.times do
	genann.train inputs[0], outputs, 0, 3
	genann.train inputs[1], outputs, 1, 3
	genann.train inputs[2], outputs, 2, 3
	genann.train inputs[3], outputs, 3, 3
end

dumpdata = genann.dump
genann = nil

genann_2 = Genann.new(dumpdata)

print sprintf("Output for [%1.f, %1.f] is %1.f.\n", inputs[0][0], inputs[0][1], genann_2.run(inputs[0]))
print sprintf("Output for [%1.f, %1.f] is %1.f.\n", inputs[1][0], inputs[1][1], genann_2.run(inputs[1]))
print sprintf("Output for [%1.f, %1.f] is %1.f.\n", inputs[2][0], inputs[2][1], genann_2.run(inputs[2]))
print sprintf("Output for [%1.f, %1.f] is %1.f.\n", inputs[3][0], inputs[3][1], genann_2.run(inputs[3]))

```
