# Train a small ANN to the XOR function using backpropagation.
assert 'example1' do
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
  assert_equal output[0][0].to_i, ann.run(input_buf[0]).unpack('E*')[0].round
  assert_equal output[1][0].to_i, ann.run(input_buf[1]).unpack('E*')[0].round
  assert_equal output[2][0].to_i, ann.run(input_buf[2]).unpack('E*')[0].round
  assert_equal output[3][0].to_i, ann.run(input_buf[3]).unpack('E*')[0].round
end

# Train a small ANN to the XOR function using random search.
assert 'example2' do
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
  weights = Array.new(ann.total_weights){ rand }
  ann.weights_load weights.pack('E*')

  err = 0.0
  last_err = 1000.0
  count = 0

  loop do
    count += 1
    if count % 1000 == 0
      # We're stuck, start over.
      weights = Array.new(ann.total_weights){ rand }
      last_err = 1000.0
    end

    save = Genann.new(2, 1, 2, 1)
    save.weights_load(ann.weights_dump)

    # Take a random guess at the ANN weights.
    ann.total_weights.times do |i|
      weights[i] += rand - 0.5
    end
    ann.weights_load weights.pack('E*')

    # See how we did.
    err = 0
    err += (ann.run(input_buf[0]).unpack('E*')[0] - output[0][0]) ** 2.0
    err += (ann.run(input_buf[1]).unpack('E*')[0] - output[1][0]) ** 2.0
    err += (ann.run(input_buf[2]).unpack('E*')[0] - output[2][0]) ** 2.0
    err += (ann.run(input_buf[3]).unpack('E*')[0] - output[3][0]) ** 2.0

    # Keep these weights if they're an improvement.
    if err < last_err
      # drop save
      last_err = err
    else
      ann = save
    end

    break unless err > 0.01
  end

  # Run the network and see what it predicts.
  # Use String#unpack for result parsing.
  assert_equal output[0][0].to_i, ann.run(input_buf[0]).unpack('E*')[0].round
  assert_equal output[1][0].to_i, ann.run(input_buf[1]).unpack('E*')[0].round
  assert_equal output[2][0].to_i, ann.run(input_buf[2]).unpack('E*')[0].round
  assert_equal output[3][0].to_i, ann.run(input_buf[3]).unpack('E*')[0].round
end


