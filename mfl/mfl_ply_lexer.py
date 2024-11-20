import ply.lex as lex

# List of token names
tokens = (
    'INT_LITERAL',
    'BOOL_LITERAL',
    'IDENTIFIER',
    'LAMBDA',
    'DOT',
    'LPAREN',
    'RPAREN',
    'LET',
    'LETREC',
    'EQUALS',
    'EQUALOP',
    'IN',
    'IF',
    'THEN',
    'ELSE',
    'PLUS',
    'MINUS',
    'TIMES',
    'DIVIDE',
    'AND',
    'OR',
    'NOT',
    'GREATER',
    'LESS',
    'LESSEQ',
    'GREATEREQ',
    'COMMA'
)

# Regular expression rules for simple tokens
t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'/'
t_DOT = r'\.'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_EQUALS = r'='
t_GREATER = r'>'
t_LESS = r'<'
t_EQUALOP = r'=='
t_LESSEQ = r'<='
t_GREATEREQ = r'>='
t_LAMBDA = r'λ'
t_ignore = ' \t'

# Define regular expression rules for keywords
reserved = {
    'let': 'LET',
    'letrec': 'LETREC',
    'in': 'IN',
    'if': 'IF',
    'then': 'THEN',
    'else': 'ELSE',
    'and': 'AND',
    'or': 'OR',
    'not': 'NOT',
    'True': 'BOOL_LITERAL',
    'False': 'BOOL_LITERAL'
}

def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    t.type = reserved.get(t.value, 'IDENTIFIER')
    if t.type == 'BOOL_LITERAL':
        t.value = t.value  # Keep the original string value
    return t

def t_INT_LITERAL(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_COMMA(t):
    r','
    return t

def t_error(t):
    print(f"Illegal character '{t.value[0]}' at line {t.lineno}")
    t.lexer.skip(1)

# Build the lexer
lexer = lex.lex()
