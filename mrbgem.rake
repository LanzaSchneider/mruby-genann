MRuby::Gem::Specification.new('mruby-genann') do |spec|
  spec.license = 'Public domain'
  spec.authors = 'Lanza Schneider'
  spec.add_test_dependency('mruby-pack', :core => 'mruby-pack')
  spec.add_test_dependency('mruby-random', :core => 'mruby-random')
end
