%option yylineno

D [0-9]
L [a-zA-Z_]

%{
#include "acc.tab.h"
%}

%%

"Scalar"    { return SCALAR; } /* Builtin types */
"Vector"    { return VECTOR; }
"Matrix"    { return MATRIX; }
"void"      { return VOID; } /* Rest of the types inherited from C */
"int"       { return INT; }
"int3"      { return INT3; }
"ScalarField" { return SCALAR; }
"VectorField" { return VECTOR; }

"Kernel"       { return KERNEL; } /* Function specifiers */
"Preprocessed" { return PREPROCESSED; }

"const"     { return CONSTANT; }
"in"        { return IN; } /* Device func storage specifiers */
"out"       { return OUT; }
"uniform"   { return UNIFORM; }

"else if"   { return ELIF; }
"if"        { return IF; }
"else"      { return ELSE; }
"for"       { return FOR; }
"while"     { return WHILE; }

"return"    { return RETURN; }

{D}+"."?{D}*[flud]? { return NUMBER; } /* Literals */
"."{D}+[flud]?      { return NUMBER; }
{L}({L}|{D})*       { return IDENTIFIER; }
\"(.)*\"            { return IDENTIFIER; } /* String */

"=="                { return LEQU; }/* Logic operations */
"&&"                { return LAND; }
"||"                { return LOR; }
"<="                { return LLEQU; }

"++"                { return INPLACE_INC; }
"--"                { return INPLACE_DEC; }

[-+*/;=\[\]{}(),\.<>] { return yytext[0]; } /* Characters */


"//".*              { /* Skip regular comments */ }
[ \t\n\v\r]+        { /* Ignore whitespace, tabs and newlines */ }
.                   { printf("unrecognized char %d: [%c]\n", *yytext, *yytext); }


%%
