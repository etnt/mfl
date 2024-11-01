; MFL to LLVM IR Compiler Output

target triple = "arm64-apple-darwin23.6.0"

declare i32 @printf(i8* nocapture readonly, ...)

; String constants for printing
@.str.int = private unnamed_addr constant [3 x i8] c"%d "
@.str.bool = private unnamed_addr constant [3 x i8] c"%s "
@.str.true = private unnamed_addr constant [5 x i8] c"true "
@.str.false = private unnamed_addr constant [6 x i8] c"false "

; -----------------------------------------------------------------------------
; This is code generated from: "let add = λx.λy.λz(x+y+z) in (add 2 3 4)"
; -----------------------------------------------------------------------------

; Define a struct to hold the captured variables for the lambda functions
%lambda_state = type { i32, i32 }  ; Holds x and y


; A type that represents a pointer to a function which:
; - takes a pointer to a lambda_state and an integer as arguments
; - returns an integer
; Function type for the final lambda that computes the result
%lambda_2_type = type i32 (%lambda_state*, i32)*

; Function type for the middle lambda that returns lambda_2
%lambda_1_type = type %lambda_2_type (%lambda_state*, i32)*


; Receives the %lambda_state (with x and y), gets the z value and performs the final addition.
define i32 @lambda_2(%lambda_state* %state, i32 %z) {
  %x_ptr = getelementptr inbounds %lambda_state, %lambda_state* %state, i32 0, i32 0
  %y_ptr = getelementptr inbounds %lambda_state, %lambda_state* %state, i32 0, i32 1
  %x = load i32, i32* %x_ptr
  %y = load i32, i32* %y_ptr
  %sum = add i32 %x, %y
  %final_sum = add i32 %sum, %z
  ret i32 %final_sum
}


; Receives the %lambda_state (with x), captures y in the same structure and returns a pointer to @lambda_2.
define %lambda_2_type @lambda_1(%lambda_state* %state, i32 %y) {
  ; Store y in the state structure
  %y_ptr = getelementptr inbounds %lambda_state, %lambda_state* %state, i32 0, i32 1
  store i32 %y, i32* %y_ptr
  ret %lambda_2_type @lambda_2
}

; Captures x in the %lambda_state and returns a pointer to @lambda_1.
define %lambda_1_type @lambda_0(i32 %x) {
  ; Allocate state on the heap instead of stack
  %state = alloca %lambda_state
  ; Store x in the state structure
  %x_ptr = getelementptr inbounds %lambda_state, %lambda_state* %state, i32 0, i32 0
  store i32 %x, i32* %x_ptr
  ret %lambda_1_type @lambda_1
}


define i32 @main() {
entry:
    ; Assigns the value 2 to x
    %x = add i32 0, 2

    ; Get the first lambda and create initial state
    %lambda1_ptr = call %lambda_1_type @lambda_0(i32 %x)
    %state = alloca %lambda_state
    %x_ptr = getelementptr inbounds %lambda_state, %lambda_state* %state, i32 0, i32 0
    store i32 %x, i32* %x_ptr

    ; Assigns the value 3 to y and get the second lambda
    %y = add i32 0, 3
    %lambda2_ptr = call %lambda_2_type %lambda1_ptr(%lambda_state* %state, i32 %y)

    ; Assigns the value 4 to z and compute final result
    %z = add i32 0, 4
    %final_result = call i32 %lambda2_ptr(%lambda_state* %state, i32 %z)

    ; Prints the result of the addition.
    call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.int, i64 0, i64 0), i32 %final_result)

    ret i32 0
}
