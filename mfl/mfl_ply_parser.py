#!/usr/bin/env python3

import ply.yacc as yacc
from mfl_ply_lexer import tokens
from mfl_ast import Int, Bool, BinOp, UnaryOp, If, Var, Function, Apply, Let, LetRec

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

def p_expr_let(p):
    '''expr : LET IDENTIFIER EQUALS expr_or_lambda IN expr'''
    p[0] = Let(Var(p[2]), p[4], p[6])

def p_expr_or_lambda(p):
    '''expr_or_lambda : expr
                       | expr_lambda'''
    p[0] = p[1]

def p_expr_letrec(p):
    '''expr : LETREC IDENTIFIER EQUALS expr_lambda IN expr'''
    p[0] = LetRec(Var(p[2]), p[4], p[6])

def p_expr_lambda(p):
    '''expr_lambda : LAMBDA IDENTIFIER DOT expr'''
    p[0] = Function(Var(p[2]), p[4])

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
