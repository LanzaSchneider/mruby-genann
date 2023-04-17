assert 'example1' do
  input = [[0, 0], [0, 1], [1, 0], [1, 1]]
  output = [[0], [1], [1], [0]]
  input_buf = input.collect {|arr| arr.pack('E*') }
  output_buf = output.collect {|arr| arr.pack('E*') }
  ann = Genann.new(2, 1, 2, 1)
  500.times do
    ann.train input_buf[0], output_buf[0], 1.0
    ann.train input_buf[1], output_buf[1], 1.0
    ann.train input_buf[2], output_buf[2], 1.0
    ann.train input_buf[3], output_buf[3], 1.0
  end
  puts sprintf('Output for [%1.f, %1.f] is %1.f.', input[0][0], input[0][1], ann.run(input_buf[0]).unpack('E*')[0])
  puts sprintf('Output for [%1.f, %1.f] is %1.f.', input[1][0], input[1][1], ann.run(input_buf[1]).unpack('E*')[0])
  puts sprintf('Output for [%1.f, %1.f] is %1.f.', input[2][0], input[2][1], ann.run(input_buf[2]).unpack('E*')[0])
  puts sprintf('Output for [%1.f, %1.f] is %1.f.', input[3][0], input[3][1], ann.run(input_buf[3]).unpack('E*')[0])
end