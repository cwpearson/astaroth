/*
    Nodes for the Abstract Syntax Tree

    Statement: syntactic unit tha expresses some action.
    May have internal components, expressions, which are evaluated

    Statements: return value
                block
*/
#include <stdlib.h>
#include <assert.h>

#define BUFFER_SIZE (4096)

#define GEN_ID(X) X
#define GEN_STR(X) #X

#define FOR_NODE_TYPES(FUNC) \
    FUNC(NODE_UNKNOWN), \
    FUNC(NODE_DEFINITION), \
    FUNC(NODE_GLOBAL_DEFINITION), \
    FUNC(NODE_DECLARATION), \
    FUNC(NODE_TYPE_QUALIFIER), \
    FUNC(NODE_TYPE_SPECIFIER), \
    FUNC(NODE_IDENTIFIER), \
    FUNC(NODE_FUNCTION_DEFINITION), \
    FUNC(NODE_FUNCTION_DECLARATION), \
    FUNC(NODE_COMPOUND_STATEMENT), \
    FUNC(NODE_FUNCTION_PARAMETER_DECLARATION), \
    FUNC(NODE_MULTIDIM_SUBSCRIPT_EXPRESSION)

/* 
// Recreating strdup is not needed when using the GNU compiler.
// Let's also just say that anything but the GNU
// compiler is NOT supported, since there are also
// some gcc-specific calls in the files generated 
// by flex and being completely compiler-independent is
// not a priority right now
#ifndef strdup 
static inline char*
strdup(const char* in)
{
    const size_t len = strlen(in) + 1;
    char* out = malloc(len);

    if (out) {
        memcpy(out, in, len);
        return out;
    } else {
        return NULL;
    }
}
#endif
*/

typedef enum {
    FOR_NODE_TYPES(GEN_ID),
    NUM_NODE_TYPES
} NodeType;

typedef struct astnode_s {
    int id;
    struct astnode_s* lhs;
    struct astnode_s* rhs;
    NodeType type;          // Type of the AST node
    char* buffer;           // Indentifiers and other strings (empty by default)

    int token;              // Type of a terminal (that is not a simple char)
    int prefix;            // Tokens. Also makes the grammar since we don't have
    int infix;             // to divide it into max two-child rules
    int postfix;           // (which makes it much harder to read)
} ASTNode;


static inline ASTNode*
astnode_create(const NodeType type, ASTNode* lhs, ASTNode* rhs)
{
    ASTNode* node = malloc(sizeof(node[0]));

    static int id_counter = 0;
    node->id     = id_counter++;
    node->type   = type;
    node->lhs    = lhs;
    node->rhs    = rhs;
    node->buffer = NULL;

    node->prefix = node->infix = node->postfix = 0;

    return node;
}

static inline void
astnode_set_buffer(const char* buffer, ASTNode* node)
{
    node->buffer = strdup(buffer);
}

static inline void
astnode_destroy(ASTNode* node)
{
    if (node->lhs)
        astnode_destroy(node->lhs);
    if (node->rhs)
        astnode_destroy(node->rhs);
    if (node->buffer)
        free(node->buffer);
    free(node);
}


extern ASTNode* root;

/*
typedef enum {
    SCOPE_BLOCK
} ScopeType;

typedef struct symbol_s {
    int type_specifier;
    char* identifier;
    int scope;
    struct symbol_s* next;
} Symbol;

extern ASTNode* symbol_table;
*/
