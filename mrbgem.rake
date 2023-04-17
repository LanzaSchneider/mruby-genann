MRuby::Gem::Specification.new('mruby-genann') do |spec|
  spec.license = 'Public domain'
  spec.authors = 'Lanza Schneider'
  spec.add_test_dependency('mruby-pack', :core => 'mruby-pack')
  spec.add_test_dependency('mruby-sprintf', :core => 'mruby-sprintf')
  spec.add_test_dependency('mruby-print', :core => 'mruby-print')
end
