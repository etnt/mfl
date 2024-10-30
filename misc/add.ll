target triple = "arm64-apple-darwin23.6.0"

declare i32 @printf(i8* nocapture readonly, ...)

; String constants for printing
@.str.int = private unnamed_addr constant [3 x i8] c"%d\00"
@.str.bool = private unnamed_addr constant [3 x i8] c"%s\00"
@.str.true = private unnamed_addr constant [5 x i8] c"true\00"
@.str.false = private unnamed_addr constant [6 x i8] c"false\00"

; -----------------------------------------------------------------------------
; This is code (hand) generated from"let add = λx.λy.(x+y) in (add 3 4)": 
; -----------------------------------------------------------------------------

; Define a struct to hold the arguments for the lambda functions
%lambda_args = type { i32, i32, i32 }

; Define add_1 to take one argument x and reference y from the lambda_args struct
define i32 @add_1(i32 %x, %lambda_args* %args) {
entry:
   ; Get y from the struct
    %y_ptr = getelementptr %lambda_args, %lambda_args* %args, i32 0, i32 1
    %y = load i32, i32* %y_ptr
    %binop_7 = add i32 %x, %y
    ret i32 %binop_7
}

; Define add_0 to take one argument which is the lambda_args struct from where
; it will get the x argument, and then call add_1 with x and the lambda_args struct.
define i32 @add_0(%lambda_args* %args) {
    ; Get x from the struct
    %x_ptr = getelementptr %lambda_args, %lambda_args* %args, i32 0, i32 0
    %x = load i32, i32* %x_ptr
    ; Call add_1 with x and the lambda args struct
    %result = call i32 @add_1(i32 %x, %lambda_args* %args)
    ret i32 %result
}

; Main function that sets up the call to add 3 and 4
define i32 @main() {
entry:
    ; Allocate a struct for the lambda arguments
    %args = alloca %lambda_args

    ; Initialize the struct with x=3
    %x_ptr = getelementptr %lambda_args, %lambda_args* %args, i32 0, i32 0
    store i32 3, i32* %x_ptr

    ; Initialize the struct with y=4
    %y_ptr = getelementptr %lambda_args, %lambda_args* %args, i32 0, i32 1
    store i32 4, i32* %y_ptr

    ; Call add_0 with the first argument (3)
    %result = call i32 @add_0(%lambda_args* %args)

    ; Print the result
    call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.int, i64 0, i64 0), i32 %result)

    ret i32 %result
}
