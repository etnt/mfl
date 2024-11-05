; ModuleID = "MFL Generated Module"
target triple = "arm64-apple-darwin23.6.0"
target datalayout = ""

%"lambda_state" = type {i32, i32, i32, i32, i32, i32, i32, i32}
declare i32 @"printf"(i8* %".1", ...)

@".str.int" = private constant [3 x i8] c"%d\00"
@".str.bool" = private constant [3 x i8] c"%s\00"
@".str.true" = private constant [5 x i8] c"true\00"
@".str.false" = private constant [6 x i8] c"false\00"
define i32 @"main"()
{
entry:
  %".2" = alloca %"lambda_state"
  %"lambda_state_0" = alloca %"lambda_state"
  ; Generated Let value: 'λx.λy.(x + y)' ,return type: i32 (%"lambda_state"*, i32)* (%"lambda_state"*, i32)*
  ; Generating function application: (add 4)
  ; Generated Apply argument: 4
  %".6" = call i32 (%"lambda_state"*, i32)* @"lambda_2"(%"lambda_state"* %"lambda_state_0", i32 4)
  ; Generating function application: ((add 4) 5)
  ; Generated Apply argument: 5
  %".9" = call i32 %".6"(%"lambda_state"* %"lambda_state_0", i32 5)
  ; Generated Let body: '((add 4) 5)' , return type: i32
  %"str_ptr" = getelementptr inbounds [3 x i8], [3 x i8]* @".str.int", i64 0, i64 0
  %".11" = call i32 (i8*, ...) @"printf"(i8* %"str_ptr", i32 %".9")
  ret i32 0
}

define i32 @"compute_1"(%"lambda_state"* %".1", i32 %".2")
{
entry:
  ; Store the captured argument in the lambda state
  %".5" = getelementptr %"lambda_state", %"lambda_state"* %".1", i32 0, i32 1
  store i32 %".2", i32* %".5"
  ; Load all the captured variable values from the lambda state
  %".8" = getelementptr %"lambda_state", %"lambda_state"* %".1", i32 0, i32 0
  %".9" = getelementptr %"lambda_state", %"lambda_state"* %".1", i32 0, i32 1
  ; Generate body: (x + y)
  %".11" = load i32, i32* %".8"
  %".12" = load i32, i32* %".9"
  %".13" = add i32 %".11", %".12"
  ret i32 %".13"
  ; Generated final function(compute_1): λy.(x + y)
}

define i32 (%"lambda_state"*, i32)* @"lambda_2"(%"lambda_state"* %".1", i32 %".2")
{
entry:
  %".4" = getelementptr %"lambda_state", %"lambda_state"* %".1", i32 0, i32 0
  store i32 %".2", i32* %".4"
  ret i32 (%"lambda_state"*, i32)* @"compute_1"
}
