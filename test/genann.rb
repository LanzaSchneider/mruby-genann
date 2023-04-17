assert 'example1' do
  input = [[0, 0], [0, 1], [1, 0], [1, 1]]
  output = [[0], [1], [1], [0]]
  input_buf = input.collect {|arr| arr.pack('E*') }
  output_buf = output.collect {|arr| arr.pack('E*') }
  ann = Genann.new(2, 1, 2, 1)
  500.times do
    ann.train input_buf[0], output_buf[0], 3
    ann.train input_buf[1], output_buf[1], 3
    ann.train input_buf[2], output_buf[2], 3
    ann.train input_buf[3], output_buf[3], 3
  end
  assert_equal output[0][0].to_i, ann.run(input_buf[0]).unpack('E*')[0].round
  assert_equal output[1][0].to_i, ann.run(input_buf[1]).unpack('E*')[0].round
  assert_equal output[2][0].to_i, ann.run(input_buf[2]).unpack('E*')[0].round
  assert_equal output[3][0].to_i, ann.run(input_buf[3]).unpack('E*')[0].round
end