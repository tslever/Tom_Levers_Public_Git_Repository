#ifndef LIST_H
#define LIST_H


typedef struct node {
    void* val;
    struct node* next;
    struct node* prev;
} node;


typedef struct list {
    int size;
    node* front;
    node* back;
} list;


void free_list(list* l);
list* make_list();
void list_insert(list*, void*);


#endif