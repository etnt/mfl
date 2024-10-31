from llvmlite import ir

def generate_llvm_ir():
    # Create the LLVM module
    module = ir.Module(name="curried_functions")
    module.triple = "arm64-apple-darwin23.6.0"

    # Basic types
    int_type = ir.IntType(32)
    int8_type = ir.IntType(8)
    int8_ptr_type = int8_type.as_pointer()

    # Add printf declaration
    printf_type = ir.FunctionType(int_type, [int8_ptr_type], var_arg=True)
    printf = ir.Function(module, printf_type, name="printf")

    # Format string for printing
    fmt_str = "%d\n\0"
    c_fmt = ir.Constant(ir.ArrayType(int8_type, len(fmt_str)), 
                       bytearray(fmt_str.encode("utf8")))
    fmt_global = ir.GlobalVariable(module, c_fmt.type, name="fmt")
    fmt_global.global_constant = True
    fmt_global.initializer = c_fmt

    # State type for captured variables
    state_type = ir.LiteralStructType([int_type, int_type])  # For x and y
    state_ptr_type = state_type.as_pointer()

    # Function types
    compute_type = ir.FunctionType(int_type, [state_ptr_type, int_type])
    lambda1_type = ir.FunctionType(compute_type.as_pointer(), [state_ptr_type, int_type])
    lambda0_type = ir.FunctionType(lambda1_type.as_pointer(), [int_type])

    # Create compute function
    compute = ir.Function(module, compute_type, name="compute")
    block = compute.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    state_ptr = compute.args[0]
    x_ptr = builder.gep(state_ptr, [ir.Constant(int_type, 0), ir.Constant(int_type, 0)])
    y_ptr = builder.gep(state_ptr, [ir.Constant(int_type, 0), ir.Constant(int_type, 1)])
    x = builder.load(x_ptr)
    y = builder.load(y_ptr)
    z = compute.args[1]
    tmp = builder.add(x, y)
    result = builder.add(tmp, z)
    builder.ret(result)

    # Create lambda1 function
    lambda1 = ir.Function(module, lambda1_type, name="lambda1")
    block = lambda1.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    state_ptr = lambda1.args[0]
    y = lambda1.args[1]
    y_ptr = builder.gep(state_ptr, [ir.Constant(int_type, 0), ir.Constant(int_type, 1)])
    builder.store(y, y_ptr)
    builder.ret(compute)

    # Create lambda0 function
    lambda0 = ir.Function(module, lambda0_type, name="lambda0")
    block = lambda0.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    state_ptr = builder.alloca(state_type)
    x = lambda0.args[0]
    x_ptr = builder.gep(state_ptr, [ir.Constant(int_type, 0), ir.Constant(int_type, 0)])
    builder.store(x, x_ptr)
    builder.ret(lambda1)

    # Create main function
    main = ir.Function(module, ir.FunctionType(int_type, []), name="main")
    block = main.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)

    # Call lambda0(2)
    x = ir.Constant(int_type, 2)
    lambda1_ptr = builder.call(lambda0, [x])

    # Create state and store x
    state_ptr = builder.alloca(state_type)
    x_ptr = builder.gep(state_ptr, [ir.Constant(int_type, 0), ir.Constant(int_type, 0)])
    builder.store(x, x_ptr)

    # Call lambda1(state, 3)
    y = ir.Constant(int_type, 3)
    compute_ptr = builder.call(lambda1_ptr, [state_ptr, y])

    # Call compute(state, 4)
    z = ir.Constant(int_type, 4)
    result = builder.call(compute_ptr, [state_ptr, z])

    # Print result
    fmt_ptr = builder.bitcast(fmt_global, int8_ptr_type)
    builder.call(printf, [fmt_ptr, result])
    builder.ret(ir.Constant(int_type, 0))

    return module

# Generate LLVM IR
module = generate_llvm_ir()
ir_code = str(module)
print("Generated LLVM IR:")
print(ir_code)

# Write to file
with open('curry_add.ll', 'w') as f:
    f.write(ir_code)

print("\nLLVM IR has been written to curry_add.ll , compile and run as: clang curry_add.ll -o curry_add && ./misc/curry_add")