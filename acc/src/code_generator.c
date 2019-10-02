/*
    Copyright (C) 2014-2019, Johannes Pekkilae, Miikka Vaeisalae.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * @file
 * \brief Brief info.
 *
 * Detailed info.
 *
 */

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "acc.tab.h"
#include "ast.h"

ASTNode* root = NULL;

// Output files
static FILE* DSLHEADER  = NULL;
static FILE* CUDAHEADER = NULL;

/*
 * =============================================================================
 * Translation
 * =============================================================================
 */
#define TRANSLATION_TABLE_SIZE (1024)
static const char* translation_table[TRANSLATION_TABLE_SIZE] = {
    [0] = NULL,
    // Control flow
    [IF]    = "if",
    [ELSE]  = "else",
    [ELIF]  = "else if",
    [WHILE] = "while",
    [FOR]   = "for",
    // Type specifiers
    [VOID]        = "void",
    [INT]         = "int",
    [INT3]        = "int3",
    [SCALAR]      = "AcReal",
    [VECTOR]      = "AcReal3",
    [MATRIX]      = "AcMatrix",
    [SCALARFIELD] = "AcReal",
    [SCALARARRAY] = "const AcReal* __restrict__",
    [COMPLEX]     = "acComplex",
    // Type qualifiers
    [KERNEL]       = "template <int step_number>  static __global__",
    [DEVICE]       = "static __device__",
    [PREPROCESSED] = "static __device__ __forceinline__",
    [CONSTANT]     = "const",
    [IN]           = "in",
    [OUT]          = "out",
    [UNIFORM]      = "uniform",
    // ETC
    [INPLACE_INC] = "++",
    [INPLACE_DEC] = "--",
    // Unary
    [','] = ",",
    [';'] = ";\n",
    ['('] = "(",
    [')'] = ")",
    ['['] = "[",
    [']'] = "]",
    ['{'] = "{\n",
    ['}'] = "}\n",
    ['='] = "=",
    ['+'] = "+",
    ['-'] = "-",
    ['/'] = "/",
    ['*'] = "*",
    ['<'] = "<",
    ['>'] = ">",
    ['!'] = "!",
    ['.'] = "."};

static const char*
translate(const int token)
{
    assert(token >= 0);
    assert(token < TRANSLATION_TABLE_SIZE);
    if (token > 0) {
        if (!translation_table[token])
            printf("ERROR: unidentified token %d\n", token);
        assert(translation_table[token]);
    }

    return translation_table[token];
}

/*
 * =============================================================================
 * Symbols
 * =============================================================================
 */
typedef enum {
    SYMBOLTYPE_FUNCTION,
    SYMBOLTYPE_FUNCTION_PARAMETER,
    SYMBOLTYPE_OTHER,
    NUM_SYMBOLTYPES
} SymbolType;

#define MAX_ID_LEN (256)
typedef struct {
    SymbolType type;
    int type_qualifier;
    int type_specifier;
    char identifier[MAX_ID_LEN];
} Symbol;

#define SYMBOL_TABLE_SIZE (65536)
static Symbol symbol_table[SYMBOL_TABLE_SIZE] = {};

#define MAX_NESTS (32)
static size_t num_symbols[MAX_NESTS] = {};
static size_t current_nest           = 0;

static Symbol*
symboltable_lookup(const char* identifier)
{
    if (!identifier)
        return NULL;

    for (size_t i = 0; i < num_symbols[current_nest]; ++i)
        if (strcmp(identifier, symbol_table[i].identifier) == 0)
            return &symbol_table[i];

    return NULL;
}

static void
add_symbol(const SymbolType type, const int tqualifier, const int tspecifier, const char* id)
{
    assert(num_symbols[current_nest] < SYMBOL_TABLE_SIZE);

    symbol_table[num_symbols[current_nest]].type           = type;
    symbol_table[num_symbols[current_nest]].type_qualifier = tqualifier;
    symbol_table[num_symbols[current_nest]].type_specifier = tspecifier;
    strcpy(symbol_table[num_symbols[current_nest]].identifier, id);

    ++num_symbols[current_nest];
}

static void
print_symbol(const size_t handle)
{
    assert(handle < SYMBOL_TABLE_SIZE);

    const char* fields[] = {
        translate(symbol_table[handle].type_qualifier),
        translate(symbol_table[handle].type_specifier),
        symbol_table[handle].identifier,
    };

    const size_t num_fields = sizeof(fields) / sizeof(fields[0]);
    for (size_t i = 0; i < num_fields; ++i)
        if (fields[i])
            printf("%s ", fields[i]);
}

static inline void
print_symbol_table(void)
{
    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
        printf("%lu: ", i);
        const char* fields[] = {translate(symbol_table[i].type_qualifier),
                                translate(symbol_table[i].type_specifier),
                                symbol_table[i].identifier};

        const size_t num_fields = sizeof(fields) / sizeof(fields[0]);
        for (size_t j = 0; j < num_fields; ++j)
            if (fields[j])
                printf("%s ", fields[j]);

        if (symbol_table[i].type == SYMBOLTYPE_FUNCTION)
            printf("(function)");
        else if (symbol_table[i].type == SYMBOLTYPE_FUNCTION_PARAMETER)
            printf("(function parameter)");
        else
            printf("(other)");
        printf("\n");
    }
}

/*
 * =============================================================================
 * Traversal state
 * =============================================================================
 */
/*
 * =============================================================================
 * AST traversal
 * =============================================================================
 */
static void
translate_latest_symbol(void)
{
    // TODO
}

static void
traverse(const ASTNode* node)
{
    // Prefix translation
    if (translate(node->prefix))
        fprintf(CUDAHEADER, "%s", translate(node->prefix));

    // Prefix logic
    if (node->type == NODE_COMPOUND_STATEMENT) {
        assert(current_nest < MAX_NESTS);

        ++current_nest;
        num_symbols[current_nest] = num_symbols[current_nest - 1];
    }

    // Traverse LHS
    if (node->lhs)
        traverse(node->lhs);

    // Infix translation
    if (translate(node->infix))
        fprintf(CUDAHEADER, "%s", translate(node->infix));

    // Infix logic
    // TODO

    // Traverse RHS
    if (node->rhs)
        traverse(node->rhs);

    // Postfix translation
    if (translate(node->postfix))
        fprintf(CUDAHEADER, "%s", translate(node->postfix));

    // Translate existing symbols
    const Symbol* symbol = symboltable_lookup(node->buffer);
    if (symbol) {
        // Uniforms
        if (symbol->type_qualifier == UNIFORM) {
            fprintf(CUDAHEADER, "DCONST(%s) ", symbol->identifier);
        }
    }

    // Add new symbols to the symbol table
    if (node->type == NODE_DECLARATION) {
        int stype;
        ASTNode* tmp = node->parent;
        while (tmp->type == NODE_DECLARATION_LIST)
            tmp = tmp->parent;

        if (tmp->type == NODE_FUNCTION_DECLARATION)
            stype = SYMBOLTYPE_FUNCTION;
        else if (tmp->type == NODE_FUNCTION_PARAMETER_DECLARATION)
            stype = SYMBOLTYPE_FUNCTION_PARAMETER;
        else
            stype = SYMBOLTYPE_OTHER;

        const ASTNode* tdeclaration = node->lhs;
        const int tqualifier        = tdeclaration->rhs ? tdeclaration->lhs->token : 0;
        const int tspecifier        = tdeclaration->rhs ? tdeclaration->rhs->token
                                                 : tdeclaration->lhs->token;

        const char* identifier = node->rhs->type == NODE_IDENTIFIER ? node->rhs->buffer
                                                                    : node->rhs->lhs->buffer;
        add_symbol(stype, tqualifier, tspecifier, identifier);

        // Translate the new symbol
        if (tqualifier == UNIFORM) {
            // Do nothing
        }
        else if (tqualifier == KERNEL) {
            fprintf(CUDAHEADER, "%s %s\n%s", //
                    translate(tqualifier), translate(tspecifier), identifier);
        }
        else if (tqualifier == DEVICE) {
            fprintf(CUDAHEADER, "%s %s\n%s", //
                    translate(tqualifier), translate(tspecifier), identifier);
        }
        else if (tqualifier == PREPROCESSED) {
            fprintf(CUDAHEADER, "%s %s\npreprocessed_%s", //
                    translate(tqualifier), translate(tspecifier), identifier);
        }
        else if (stype == SYMBOLTYPE_FUNCTION_PARAMETER) {
            tmp = tmp->parent;
            assert(tmp->type = NODE_FUNCTION_DECLARATION);
            const Symbol* parent_function = symboltable_lookup(tmp->lhs->rhs->buffer);
            if (parent_function->type_qualifier == DEVICE)
                fprintf(CUDAHEADER, "%s %s\ndeviceparam_%s", //
                        translate(tqualifier), translate(tspecifier), identifier);
            else if (parent_function->type_qualifier == PREPROCESSED)
                fprintf(CUDAHEADER, "%s %s\npreprocessedparam_%s", //
                        translate(tqualifier), translate(tspecifier), identifier);
            else
                fprintf(CUDAHEADER, "%s %s\notherparam_%s", //
                        translate(tqualifier), translate(tspecifier), identifier);
        }
        else { // Do a regular translation
            // fprintf(CUDAHEADER, "%s %s %s", //
            //        translate(tqualifier), translate(tspecifier), identifier);
        }
    }

    // Postfix logic
    if (node->type == NODE_COMPOUND_STATEMENT) {
        assert(current_nest > 0);
        --current_nest;
        printf("Dropped rest of the symbol table, from %lu to %lu\n", num_symbols[current_nest + 1],
               num_symbols[current_nest]);
    }
}

static void
generate_preprocessed_structures(void)
{
    // TODO
}

static void
generate_header(void)
{
    fprintf(DSLHEADER, "#pragma once\n");

    // Int params
    fprintf(DSLHEADER, "#define AC_FOR_USER_INT_PARAM_TYPES(FUNC)");
    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
        if (symbol_table[i].type_specifier == INT && symbol_table[i].type_qualifier == UNIFORM) {
            fprintf(DSLHEADER, "\\\nFUNC(%s),", symbol_table[i].identifier);
        }
    }
    fprintf(DSLHEADER, "\n\n");

    // Int3 params
    fprintf(DSLHEADER, "#define AC_FOR_USER_INT3_PARAM_TYPES(FUNC)");
    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
        if (symbol_table[i].type_specifier == INT3 && symbol_table[i].type_qualifier == UNIFORM) {
            fprintf(DSLHEADER, "\\\nFUNC(%s),", symbol_table[i].identifier);
        }
    }
    fprintf(DSLHEADER, "\n\n");

    // Scalar params
    fprintf(DSLHEADER, "#define AC_FOR_USER_REAL_PARAM_TYPES(FUNC)");
    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
        if (symbol_table[i].type_specifier == SCALAR && symbol_table[i].type_qualifier == UNIFORM) {
            fprintf(DSLHEADER, "\\\nFUNC(%s),", symbol_table[i].identifier);
        }
    }
    fprintf(DSLHEADER, "\n\n");

    // Vector params
    fprintf(DSLHEADER, "#define AC_FOR_USER_REAL3_PARAM_TYPES(FUNC)");
    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
        if (symbol_table[i].type_specifier == VECTOR && symbol_table[i].type_qualifier == UNIFORM) {
            fprintf(DSLHEADER, "\\\nFUNC(%s),", symbol_table[i].identifier);
        }
    }
    fprintf(DSLHEADER, "\n\n");

    // Scalar fields
    fprintf(DSLHEADER, "#define AC_FOR_VTXBUF_HANDLES(FUNC)");
    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
        if (symbol_table[i].type_specifier == SCALARFIELD &&
            symbol_table[i].type_qualifier == UNIFORM) {
            fprintf(DSLHEADER, "\\\nFUNC(%s),", symbol_table[i].identifier);
        }
    }
    fprintf(DSLHEADER, "\n\n");

    // Scalar arrays
    fprintf(DSLHEADER, "#define AC_FOR_SCALARARRAY_HANDLES(FUNC)");
    for (size_t i = 0; i < num_symbols[current_nest]; ++i) {
        if (symbol_table[i].type_specifier == SCALARARRAY &&
            symbol_table[i].type_qualifier == UNIFORM) {
            fprintf(DSLHEADER, "\\\nFUNC(%s),", symbol_table[i].identifier);
        }
    }
    fprintf(DSLHEADER, "\n\n");
}

static void
generate_library_hooks(void)
{
    for (int i = 0; i < num_symbols[current_nest]; ++i) {
        if (symbol_table[i].type_qualifier == KERNEL && symbol_table[i].type_qualifier == UNIFORM) {
            fprintf(CUDAHEADER, "GEN_DEVICE_FUNC_HOOK(%s)\n", symbol_table[i].identifier);
        }
    }
}

int
main(int argc, char** argv)
{
    root = astnode_create(NODE_UNKNOWN, NULL, NULL);

    const int retval = yyparse();
    if (retval) {
        fprintf(stderr, "COMPILATION FAILED\n");
        return EXIT_FAILURE;
    }

    DSLHEADER  = fopen("user_defines.h", "w+");
    CUDAHEADER = fopen("user_kernels.h", "w+");
    assert(DSLHEADER);
    assert(CUDAHEADER);

    traverse(root);
    generate_header();
    generate_library_hooks();

    print_symbol_table();

    // Cleanup
    fclose(DSLHEADER);
    fclose(CUDAHEADER);
    astnode_destroy(root);
    fprintf(stdout, "COMPILATION SUCCESS\n");
    return EXIT_SUCCESS;
}
