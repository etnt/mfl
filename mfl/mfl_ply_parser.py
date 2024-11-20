#!/usr/bin/env python3

import ply.yacc as yacc
from mfl_ply_lexer import tokens
from mfl_ast import Int, Bool, BinOp, UnaryOp, If, Var, Function, Apply, Lets, Let, LetRec, LetBinding

precedence = (
    ('left', 'PLUS', 'MINUS'),
    ('left', 'TIMES', 'DIVIDE'),
)

def p_prog(p):
    '''prog : expr'''
    p[0] = p[1]

def p_expr_binop(p):
    '''expr : expr PLUS expr
           | expr MINUS expr
           | expr TIMES expr
           | expr DIVIDE expr
           | expr GREATER expr
           | expr LESS expr
           | expr EQUALOP expr
           | expr LESSEQ expr
           | expr GREATEREQ expr
           | expr AND expr
           | expr OR expr'''
    p[0] = BinOp(p[2], p[1], p[3])

def p_expr_group(p):
    '''expr : LPAREN expr RPAREN'''
    p[0] = p[2]

def p_expr_number(p):
    '''expr : INT_LITERAL'''
    p[0] = Int(p[1])

def p_expr_bool(p):
    '''expr : BOOL_LITERAL'''
    # Convert string 'True'/'False' to bool value
    p[0] = Bool(p[1] == 'True')

def p_expr_if(p):
    '''expr : IF expr THEN expr ELSE expr'''
    p[0] = If(p[2], p[4], p[6])

def p_expr_var(p):
    '''expr : IDENTIFIER'''
    p[0] = Var(p[1])

def p_expr_let_multiple(p):
    '''expr : LET let_bindings IN expr'''
    # Convert multiple bindings into nested Let expressions
    bindings = p[2]
    body = p[4]

    # Start from the last binding and work backwards
    result = body
    for binding in reversed(bindings):
        result = Let(binding.name, binding.value, result)

    p[0] = result

def p_let_bindings(p):
    '''let_bindings : let_binding
                    | let_binding COMMA let_bindings'''
    if len(p) == 2:
        p[0] = [p[1]]  # Single let binding wrapped in list
    elif type(p[3]) is list:
        p[0] = [p[1]] + p[3]  # Concatenate lists of bindings
    else:
        p[0] = [p[1]] + [p[3]]  # Concatenate lists of bindings

def p_let_binding(p):
    '''let_binding : IDENTIFIER EQUALS expr_or_lambda'''
    p[0] = LetBinding(Var(p[1]), p[3])

def p_expr_or_lambda(p):
    '''expr_or_lambda : expr
                     | lambda_expr'''
    p[0] = p[1]

def p_lambda_expr(p):
    '''lambda_expr : LAMBDA IDENTIFIER DOT expr_or_lambda'''
    p[0] = Function(Var(p[2]), p[4])

def p_expr_letrec(p):
    '''expr : LETREC IDENTIFIER EQUALS lambda_expr IN expr'''
    p[0] = LetRec(Var(p[2]), p[4], p[6])

def p_expr_app(p):
    '''expr : expr expr %prec TIMES'''
    p[0] = Apply(p[1], p[2])

def p_expr_not(p):
    '''expr : NOT expr'''
    p[0] = UnaryOp('!', p[2])

def p_error(p):
    if p:
        raise ValueError(f"Syntax error at '{p.value}'")
    else:
        raise ValueError("Syntax error at EOF")


parser = yacc.yacc()

def parse(text):
    return parser.parse(text)

if __name__ == '__main__':
    expr = input('Enter an expression: ')
    ast = parse(expr)
    print(f"AST(raw): {ast.raw_structure()}")
