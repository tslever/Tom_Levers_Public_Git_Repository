#include "pch.h"
#include "list.h"
#include <corecrt_malloc.h>
#include "utils.h"


void free_node(node* n)
{
    node* next;
    while (n) {
        next = n->next;
        free(n);
        n = next;
    }
}


void free_list(list* l)
{
    free_node(l->front);
    free(l);
}


list* make_list()
{
    list* l = (list*)xmalloc(sizeof(list));
    l->size = 0;
    l->front = 0;
    l->back = 0;
    return l;
}


void list_insert(list* l, void* val)
{
    node* newnode = (node*)xmalloc(sizeof(node));
    newnode->val = val;
    newnode->next = 0;

    if (!l->back) {
        l->front = newnode;
        newnode->prev = 0;
    }
    else {
        l->back->next = newnode;
        newnode->prev = l->back;
    }
    l->back = newnode;
    ++l->size;
}