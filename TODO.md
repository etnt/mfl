# TODO

Just some ideas of what could be fun to implement.

## letrec

```ml
letrec fibonacci =
  λx.if x = 0 then 0
     else if x = 1 then 1
          else fibonacci (x - 1) + fibonacci (x - 2) in (fibonacci 5)
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

> python3 ./mfl/mfl.py -a "let y = λf.(λx.f (x x)) (λx.f (x x)) in let fibonacci = λf.λx.if x = 0 then 0 else if x = 1 then 1 else f (x - 1) + f (x - 2) in ((y fibonacci) 5)"
Syntax error at '='
yntax error at 'in'
```
 