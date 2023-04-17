#include <mruby.h>
#include <mruby/data.h>
#include <mruby/class.h>
#include <mruby/string.h>
#include <mruby/presym.h>

#include <genann.h>
#include <string.h>

enum genann_actfunc
{
  GENANN_ACTFUNC_SIGMOID,
  GENANN_ACTFUNC_SIGMOID_CACHED,
  GENANN_ACTFUNC_THRESHOLD,
  GENANN_ACTFUNC_LINEAR,
  GENANN_ACTFUNC_MAX
};

static void gn_change_actfunc(mrb_int type, genann_actfun *target)
{
  switch (type)
  {
  case GENANN_ACTFUNC_SIGMOID:
    *target = genann_act_sigmoid;
  break;
  case GENANN_ACTFUNC_SIGMOID_CACHED:
    *target = genann_act_sigmoid_cached;
  break;
  case GENANN_ACTFUNC_THRESHOLD:
    *target = genann_act_threshold;
  break;
  case GENANN_ACTFUNC_LINEAR:
    *target = genann_act_linear;
  break;
  }
}

static void
gn_data_free(void *ptr)
{
  if (ptr)
    genann_free((genann *) ptr);
}

static const struct mrb_data_type gn_type = { "Genann", gn_data_free };

static struct genann *
ann_get_ptr(mrb_state *mrb, mrb_value gn)
{
  genann *ann;

  ann = DATA_GET_PTR(mrb, gn, &gn_type, genann);
  if (!ann) {
    mrb_raise(mrb, E_ARGUMENT_ERROR, "uninitialized Genann");
  }
  return ann;
}

static mrb_value
gn_init(mrb_state *mrb, mrb_value self)
{
  genann *ann;
  mrb_int inputs, hidden_layers, hidden, outputs;
  mrb_int activation_hidden = -1;
  mrb_int activation_output = -1;
  mrb_get_args(mrb, "iiii|ii", &inputs, &hidden_layers, &hidden, &outputs, &activation_hidden, &activation_output);
  ann = genann_init(inputs, hidden_layers, hidden, outputs);
  if (!ann)
    mrb_raise(mrb, E_ARGUMENT_ERROR, "cannot initialize genann with given arguments");
  gn_change_actfunc(activation_hidden, &ann->activation_hidden);
  gn_change_actfunc(activation_output, &ann->activation_output);
  mrb_data_init(self, ann, &gn_type);
  return self;
}

static mrb_value
gn_train(mrb_state *mrb, mrb_value self)
{
  genann *ann;
  mrb_value buf_inputs;
  mrb_value buf_desired_outputs;
  mrb_float learning_rate;
  mrb_get_args(mrb, "SSf", &buf_inputs, &buf_desired_outputs, &learning_rate);
  ann = ann_get_ptr(mrb, self);
  if (sizeof(double) * ann->inputs != RSTRING_LEN(buf_inputs))
    mrb_raise(mrb, E_ARGUMENT_ERROR, "buf_inputs must be a pack of %d double-precision floating-point numbers", (mrb_int) ann->inputs);
  if (sizeof(double) * ann->outputs != RSTRING_LEN(buf_desired_outputs))
    mrb_raise(mrb, E_ARGUMENT_ERROR, "buf_desired_outputs must be a pack of %d double-precision floating-point numbers", (mrb_int) ann->outputs);
  genann_train(ann, (const double *) RSTRING_PTR(buf_inputs), (const double *) RSTRING_PTR(buf_desired_outputs), (double) learning_rate);
  return self;
}

static mrb_value
gn_run(mrb_state *mrb, mrb_value self)
{
  genann *ann;
  mrb_value buf_inputs;
  mrb_get_args(mrb, "S", &buf_inputs);
  ann = ann_get_ptr(mrb, self);
  if (sizeof(double) * ann->inputs != RSTRING_LEN(buf_inputs))
    mrb_raise(mrb, E_ARGUMENT_ERROR, "buf_inputs must be a pack of %d double-precision floating-point numbers", (mrb_int) ann->inputs);
  return mrb_str_new(mrb, (const char *) genann_run(ann, (const double *) RSTRING_PTR(buf_inputs)), sizeof(double) * ann->outputs);
}

static mrb_value gn_get_inputs(mrb_state *mrb, mrb_value self) { return mrb_fixnum_value(ann_get_ptr(mrb, self)->inputs); }
static mrb_value gn_get_hidden_layers(mrb_state *mrb, mrb_value self) { return mrb_fixnum_value(ann_get_ptr(mrb, self)->hidden_layers); }
static mrb_value gn_get_hidden(mrb_state *mrb, mrb_value self) { return mrb_fixnum_value(ann_get_ptr(mrb, self)->hidden); }
static mrb_value gn_get_outputs(mrb_state *mrb, mrb_value self) { return mrb_fixnum_value(ann_get_ptr(mrb, self)->outputs); }
static mrb_value gn_get_total_weights(mrb_state *mrb, mrb_value self) { return mrb_fixnum_value(ann_get_ptr(mrb, self)->total_weights); }
static mrb_value gn_get_total_neurons(mrb_state *mrb, mrb_value self) { return mrb_fixnum_value(ann_get_ptr(mrb, self)->total_neurons); }

static mrb_value 
gn_weights_dump(mrb_state *mrb, mrb_value self)
{
  genann *ann = ann_get_ptr(mrb, self);
  return mrb_str_new(mrb, (const char *) ann->weight, sizeof(double) * ann->total_weights);
}

static mrb_value
gn_weights_load(mrb_state *mrb, mrb_value self)
{
  genann *ann;
  mrb_value buf_weights;
  mrb_get_args(mrb, "S", &buf_weights);
  ann = ann_get_ptr(mrb, self);
  if (sizeof(double) * ann->total_weights != RSTRING_LEN(buf_weights))
    mrb_raise(mrb, E_ARGUMENT_ERROR, "buf_weights must be a pack of %d double-precision floating-point numbers", (mrb_int) ann->total_weights);
  memcpy(ann->weight, RSTRING_PTR(buf_weights), sizeof(double) * ann->total_weights);
  return self;
}

void 
mrb_mruby_genann_gem_init(mrb_state* mrb)
{
  struct RClass *gn_c;
  gn_c = mrb_define_class_id(mrb, MRB_SYM(Genann), mrb->object_class);
  MRB_SET_INSTANCE_TT(gn_c, MRB_TT_CDATA);

  mrb_define_method_id(mrb, gn_c, MRB_SYM(initialize), gn_init, MRB_ARGS_ARG(4, 2));
  mrb_define_method_id(mrb, gn_c, MRB_SYM(train), gn_train, MRB_ARGS_REQ(3));
  mrb_define_module_id(mrb, gn_c, MRB_SYM(run), gn_run, MRB_ARGS_REQ(1));

  mrb_define_module_id(mrb, gn_c, MRB_SYM(inputs), gn_get_inputs, MRB_ARGS_NONE());
  mrb_define_module_id(mrb, gn_c, MRB_SYM(hidden_layers), gn_get_hidden_layers, MRB_ARGS_NONE());
  mrb_define_module_id(mrb, gn_c, MRB_SYM(hidden), gn_get_hidden, MRB_ARGS_NONE());
  mrb_define_module_id(mrb, gn_c, MRB_SYM(outputs), gn_get_outputs, MRB_ARGS_NONE());
  mrb_define_module_id(mrb, gn_c, MRB_SYM(total_weights), gn_get_total_weights, MRB_ARGS_NONE());
  mrb_define_module_id(mrb, gn_c, MRB_SYM(total_neurons), gn_get_total_neurons, MRB_ARGS_NONE());

  mrb_define_module_id(mrb, gn_c, MRB_SYM(weights_dump), gn_weights_dump, MRB_ARGS_NONE());
  mrb_define_module_id(mrb, gn_c, MRB_SYM(weights_load), gn_weights_load, MRB_ARGS_REQ(1));
  
  mrb_define_const_id(mrb, gn_c, MRB_SYM(ACTFUNC_SIGMOID), mrb_fixnum_value(GENANN_ACTFUNC_SIGMOID));
  mrb_define_const_id(mrb, gn_c, MRB_SYM(ACTFUNC_SIGMOID_CACHED), mrb_fixnum_value(GENANN_ACTFUNC_SIGMOID_CACHED));
  mrb_define_const_id(mrb, gn_c, MRB_SYM(ACTFUNC_THRESHOLD), mrb_fixnum_value(GENANN_ACTFUNC_THRESHOLD));
  mrb_define_const_id(mrb, gn_c, MRB_SYM(ACTFUNC_LINEAR), mrb_fixnum_value(GENANN_ACTFUNC_LINEAR));
}

void 
mrb_mruby_genann_gem_final(mrb_state* mrb)
{
}
