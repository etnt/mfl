# TODO

Just some ideas of what could be fun to implement.

## letrec

```ml
letrec gcd = 
  λn.λm.(if (n == m) then n 
          else (if (n < m) then (gcd n (m - n)) 
            else (gcd (n - m) m))) 
  in (gcd 126 70)"
```

```ml
letrec ackerman =
  λm.λn.if m = 0 then n + 1
    else if n = 0 then ackerman (m - 1) 1
      else ackerman (m - 1) (ackerman m (n - 1)) in (ackerman 3 10)
 ```

 ## Y-combinator

 ```ml
 let y = λf.(λx.f (x x)) (λx.f (x x)) in
    let fibonacci =
        λf.λx.if x = 0 then 0
        else if x = 1 then 1
          else f (x - 1) + f (x - 2) in ((y fibonacci) 5)

python3 ./mfl/mfl.py -f "let y = (λf.(λx.f (x x)) (λx.f (x x))) in let fibonacci = λf.λx.(if (x == 0) then 0 else (if (x == 1) then 1 else (f (x - 1) + f (x - 2)))) in ((y fibonacci) 5)"                                          
Successfully parsed!
AST(pretty): let y = λf.(λx.(f (x x)) λx.(f (x x))) in let fibonacci = λf.λx.if (x == 0) then 0 else if (x == 1) then 1 else (((f (x - 1)) + f) (x - 2)) in ((y fibonacci) 5)
AST(raw): Let(Var("y"), Function(Var("f"), Apply(Function(Var("x"), Apply(Var("f"), Apply(Var("x"), Var("x")))), Function(Var("x"), Apply(Var("f"), Apply(Var("x"), Var("x")))))), Let(Var("fibonacci"), Function(Var("f"), Function(Var("x"), If(BinOp("==", Var("x"), Int(0)), Int(0), If(BinOp("==", Var("x"), Int(1)), Int(1), Apply(BinOp("+", Apply(Var("f"), BinOp("-", Var("x"), Int(1))), Var("f")), BinOp("-", Var("x"), Int(2))))))), Apply(Apply(Var("y"), Var("fibonacci")), Int(5))))
Error during type checking: Recursive type found: a6 and ->(a6, a12)
```
 