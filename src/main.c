#include <mruby.h>
#include <mruby/data.h>
#include <mruby/array.h>
#include <mruby/class.h>
#include <mruby/value.h>
#include <mruby/string.h>
#include <mruby/numeric.h>
#include <mruby/variable.h>

#include <genann.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define SINGLE_DUMP_DATA_BYTE 30

static void mrb_genann_free(mrb_state *mrb, void *ptr) {
	if ( ptr ) genann_free((genann*)ptr);
}
struct mrb_data_type mrb_genann_type = { "Genann", mrb_genann_free };

static void mrb_genann_array_free(mrb_state *mrb, void *ptr) {
	if ( ptr ) free((double*)ptr);
}
struct mrb_data_type mrb_genann_array_type = { "Genann/Array", mrb_genann_array_free };

// =======================================================
// Genann
// =======================================================
static mrb_value mrb_genann_initialize(mrb_state* mrb, mrb_value self) {
	genann* gnn = NULL;
	mrb_int inputs, hidden_layers, hidden, outputs;
	mrb_value dumpdata;
	
	int argc = mrb_get_argc(mrb);
	
	if (argc == 4 && 4 == mrb_get_args(mrb, "iiii", &inputs, &hidden_layers, &hidden, &outputs)) {
		gnn = genann_init(inputs, hidden_layers, hidden, outputs);
	} else if (argc == 1 && mrb_get_args(mrb, "S", &dumpdata)) {
		int datasize = RSTRING_LEN(dumpdata);
		if (datasize > sizeof(int32_t) * 4) {
			char* base = RSTRING_PTR(dumpdata);
			int32_t* int_base = (int32_t*)base;
			gnn = genann_init(int_base[0], int_base[1], int_base[2], int_base[3]);
			if ( gnn ) {
				if ( datasize < sizeof(int32_t) * 4 + gnn->total_weights * SINGLE_DUMP_DATA_BYTE ) {
					genann_free(gnn);
					gnn = NULL;
				} else {
					base += sizeof(int32_t) * 4;
					for (int32_t i = 0; i < gnn->total_weights; i++) {
						sscanf(base, "%le", gnn->weight + i);
						base += SINGLE_DUMP_DATA_BYTE;
					}
				}
			}
		}
		
	}
	if (gnn) {
		DATA_PTR(self) = gnn;
		DATA_TYPE(self) = &mrb_genann_type;
	} else {
		mrb_raise(mrb, E_RUNTIME_ERROR, "Genann initialize failed(need args: | inputs(fixnum), hidden_layers(fixnum), hidden(fixnum), outputs(fixnum) | or dumpdata(string)).");
	}
	return mrb_nil_value();
}

static mrb_value mrb_genann_train(mrb_state* mrb, mrb_value self) {
	genann* gnn = (genann*)DATA_PTR(self);
	mrb_value inputs, desired_outputs;
	mrb_int desired_output_index;
	mrb_float learning_rate;
	if ( gnn && (4 == mrb_get_args(mrb, "ooif", &inputs, &desired_outputs, &desired_output_index, &learning_rate))) {
		if ((DATA_TYPE(inputs) == &mrb_genann_array_type) &&
			DATA_PTR(inputs) &&
			(DATA_TYPE(desired_outputs) == &mrb_genann_array_type) &&
			DATA_PTR(desired_outputs)) {
			genann_train(gnn, (double const*)DATA_PTR(inputs), (double const*)(DATA_PTR(desired_outputs)) + desired_output_index, (double)learning_rate);
		} else if (mrb_array_p(inputs) && mrb_array_p(desired_outputs) && desired_output_index >= 0 && desired_output_index < RARRAY_LEN(desired_outputs) ) {
			
			double desired_output = (double)mrb_as_float(mrb, RARRAY_PTR(desired_outputs)[desired_output_index]);
			double* inputs_array = (double*)malloc(sizeof(double) * RARRAY_LEN(inputs));
			for (int i = 0; i < RARRAY_LEN(inputs); i++) {
				inputs_array[i] = (double)mrb_as_float(mrb, RARRAY_PTR(inputs)[i]);
			}
			genann_train(gnn, inputs_array, &desired_output, (double)learning_rate);
			free(inputs_array);
		}
	} else {
		mrb_raise(mrb, E_RUNTIME_ERROR, "Genann train failed(need args: inputs(Genann::Array), desired_outputs(Genann::Array), desired_output_index(fixnum), learning_rate(float)) | or inputs(Array), desired_outputs(Array), desired_output_index(fixnum), learning_rate(float).");
	}
	return mrb_nil_value();
}

static mrb_value mrb_genann_run(mrb_state* mrb, mrb_value self) {
	genann* gnn = (genann*)DATA_PTR(self);
	mrb_value inputs;
	if ( gnn && mrb_get_args(mrb, "o", &inputs) ) {
		double out = -1;
		if ((DATA_TYPE(inputs) == &mrb_genann_array_type) && DATA_PTR(inputs)) {
			out = *genann_run(gnn, (double const*)DATA_PTR(inputs));
		} else if (mrb_array_p(inputs)) {
			double* inputs_array = (double*)malloc(sizeof(double) * RARRAY_LEN(inputs));
			for (int i = 0; i < RARRAY_LEN(inputs); i++) {
				inputs_array[i] = (double)mrb_as_float(mrb, RARRAY_PTR(inputs)[i]);
			}
			out = *genann_run(gnn, inputs_array);
			free(inputs_array);
		}
		return mrb_float_value(mrb, (mrb_float)out);
	} else {
		mrb_raise(mrb, E_RUNTIME_ERROR, "Genann run failed(need args: inputs(Genann::Array) | or inputs(Array).");
	}
	return mrb_nil_value();
}

static mrb_value mrb_genann_train_multi(mrb_state* mrb, mrb_value self) {
	genann* gnn = (genann*)DATA_PTR(self);
	mrb_value inputs, desired_outputs;
	mrb_float learning_rate;
	if ( gnn && (3 == mrb_get_args(mrb, "oof", &inputs, &desired_outputs, &learning_rate))) {
		if ((DATA_TYPE(inputs) == &mrb_genann_array_type) &&
			DATA_PTR(inputs) &&
			(DATA_TYPE(desired_outputs) == &mrb_genann_array_type) &&
			DATA_PTR(desired_outputs)) {
			genann_train(gnn, (double const*)DATA_PTR(inputs), (double const*)(DATA_PTR(desired_outputs)), (double)learning_rate);
		} else if (mrb_array_p(inputs) && mrb_array_p(desired_outputs)) {
			double* desired_outputs_array = (double*)malloc(sizeof(double) * RARRAY_LEN(desired_outputs));
			for (int i = 0; i < RARRAY_LEN(desired_outputs); i++) {
				desired_outputs_array[i] = (double)mrb_as_float(mrb, RARRAY_PTR(desired_outputs)[i]);
			}
			double* inputs_array = (double*)malloc(sizeof(double) * RARRAY_LEN(inputs));
			for (int i = 0; i < RARRAY_LEN(inputs); i++) {
				inputs_array[i] = (double)mrb_as_float(mrb, RARRAY_PTR(inputs)[i]);
			}
			genann_train(gnn, inputs_array, desired_outputs_array, (double)learning_rate);
			free(inputs_array);
			free(desired_outputs_array);
		}
	} else {
		mrb_raise(mrb, E_RUNTIME_ERROR, "Genann train_multi failed(need args: inputs(Genann::Array), desired_outputs(Genann::Array), learning_rate(float)) | or inputs(Array), desired_outputs(Array), learning_rate(float).");
	}
	return mrb_nil_value();
}

static mrb_value mrb_genann_run_multi(mrb_state* mrb, mrb_value self) {
	genann* gnn = (genann*)DATA_PTR(self);
	mrb_value inputs;
	if ( gnn && mrb_get_args(mrb, "o", &inputs) ) {
		mrb_value result = mrb_ary_new(mrb);
		if ((DATA_TYPE(inputs) == &mrb_genann_array_type) && DATA_PTR(inputs)) {
			const double* out = genann_run(gnn, (double const*)DATA_PTR(inputs));
			for (int i = 0; i < gnn->outputs; i++) {
				mrb_ary_push(mrb, result, mrb_float_value(mrb, (mrb_float)out[i]));
			}
		} else if (mrb_array_p(inputs)) {
			double* inputs_array = (double*)malloc(sizeof(double) * RARRAY_LEN(inputs));
			for (int i = 0; i < RARRAY_LEN(inputs); i++) {
				inputs_array[i] = (double)mrb_as_float(mrb, RARRAY_PTR(inputs)[i]);
			}
			const double* out = genann_run(gnn, inputs_array);
			for (int i = 0; i < gnn->outputs; i++) {
				mrb_ary_push(mrb, result, mrb_float_value(mrb, (mrb_float)out[i]));
			}
			free(inputs_array);
		}
		return result;
	} else {
		mrb_raise(mrb, E_RUNTIME_ERROR, "Genann run_multi failed(need args: inputs(Genann::Array) | or inputs(Array).");
	}
	return mrb_nil_value();
}

static mrb_value mrb_genann_dump(mrb_state* mrb, mrb_value self) {
	genann* gnn = (genann*)DATA_PTR(self);
	if ( gnn ) {
		int size = sizeof(int32_t) * 4 + 
		gnn->total_weights * SINGLE_DUMP_DATA_BYTE;
		mrb_value result = mrb_str_buf_new(mrb, size);
		mrb_str_resize(mrb, result, size);
		char* base = RSTRING_PTR(result);
		{
			int32_t* int_base = (int32_t*)base;
			int_base[0] = gnn->inputs;
			int_base[1] = gnn->hidden_layers;
			int_base[2] = gnn->hidden;
			int_base[3] = gnn->outputs;
		}
		base += sizeof(int32_t) * 4;
		for (int32_t i = 0; i < gnn->total_weights; i++) {
			sprintf(base, "%.20e", gnn->weight[i]);
			base += SINGLE_DUMP_DATA_BYTE;
		}
		return result;
	}
	return mrb_str_buf_new(mrb, 0);
}

// =======================================================
// Genann Array
// =======================================================
static mrb_value mrb_genann_array_initialize(mrb_state* mrb, mrb_value self) {
	double* array = NULL;
	mrb_value arg;
	if (mrb_get_args(mrb, "o", &arg)) {
		if ( mrb_array_p(arg) ) {
			unsigned len = RARRAY_LEN(arg);
			mrb_value* base = RARRAY_PTR(arg);
			array = (double*)malloc(sizeof(double) * len);
			for (unsigned i = 0; i < len; i++) {
				array[i] = mrb_as_float(mrb, base[i]);
			}
			mrb_iv_set(mrb, self, mrb_intern_lit(mrb, "__size__"), mrb_fixnum_value((mrb_int)len));
		} else if (mrb_fixnum_p(arg)) {
			array = (double*)malloc(sizeof(double) * mrb_fixnum(arg));
			mrb_iv_set(mrb, self, mrb_intern_lit(mrb, "__size__"), arg);
		}
	}
	if (array) {
		DATA_PTR(self) = array;
		DATA_TYPE(self) = &mrb_genann_array_type;
	} else {
		mrb_raise(mrb, E_RUNTIME_ERROR, "Genann::Array initialize failed(need args: | size(fixnum) | or source(Array).");
	}
	return mrb_nil_value();
}

static mrb_value mrb_genann_array_get(mrb_state* mrb, mrb_value self) {
	double* array = (double*)DATA_PTR(self);
	mrb_int value = 0;
	if ( array && mrb_get_args(mrb, "i", &value) ) {
		return mrb_float_value(mrb, (float)array[value]);
	}
	return mrb_nil_value();
}

static mrb_value mrb_genann_array_set(mrb_state* mrb, mrb_value self) {
	double* array = (double*)DATA_PTR(self);
	mrb_int index;
	mrb_float value;
	if ( array && (2 == mrb_get_args(mrb, "if", &index, &value)) ) {
		array[index] = (double)value;
	}
	return mrb_nil_value();
}

static mrb_value mrb_genann_array_size(mrb_state* mrb, mrb_value self) {
	return mrb_iv_get(mrb, self, mrb_intern_lit(mrb, "__size__"));
}

void mrb_mruby_genann_gem_init(mrb_state* mrb) {
	struct RClass* genann_class = mrb_define_class(mrb, "Genann", mrb->object_class);
	
	MRB_SET_INSTANCE_TT(genann_class, MRB_TT_DATA);

	mrb_define_method(mrb, genann_class, "initialize", mrb_genann_initialize, MRB_ARGS_ANY());
	mrb_define_method(mrb, genann_class, "train", mrb_genann_train, MRB_ARGS_REQ(4));
	mrb_define_method(mrb, genann_class, "run", mrb_genann_run, MRB_ARGS_REQ(1));
	mrb_define_method(mrb, genann_class, "train_multi", mrb_genann_train_multi, MRB_ARGS_REQ(3));
	mrb_define_method(mrb, genann_class, "run_multi", mrb_genann_run_multi, MRB_ARGS_REQ(1));
	mrb_define_method(mrb, genann_class, "dump", mrb_genann_dump, MRB_ARGS_NONE());
	
	struct RClass* genann_array_class = mrb_define_class_under(mrb, genann_class, "Array", mrb->object_class);
	
	MRB_SET_INSTANCE_TT(genann_array_class, MRB_TT_DATA);
	
	mrb_define_method(mrb, genann_array_class, "initialize", mrb_genann_array_initialize, MRB_ARGS_REQ(1));
	mrb_define_method(mrb, genann_array_class, "[]", mrb_genann_array_get, MRB_ARGS_REQ(1));
	mrb_define_method(mrb, genann_array_class, "[]=", mrb_genann_array_set, MRB_ARGS_REQ(2));
	mrb_define_method(mrb, genann_array_class, "size", mrb_genann_array_size, MRB_ARGS_NONE());
	mrb_define_method(mrb, genann_array_class, "length", mrb_genann_array_size, MRB_ARGS_NONE());
}

void mrb_mruby_genann_gem_final(mrb_state* mrb) {}
