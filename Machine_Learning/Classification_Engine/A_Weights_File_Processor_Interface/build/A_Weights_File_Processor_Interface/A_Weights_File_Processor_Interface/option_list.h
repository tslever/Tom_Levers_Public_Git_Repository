#ifndef OPTION_LIST_H
#define OPTION_LIST_H


#include "list.h"


typedef struct {
    char* key;
    char* val;
    int used;
} kvp;


char* option_find(list* l, char* key);
float option_find_float_quiet(list* l, char* key, float def);
int option_find_int_quiet(list* l, char* key, int def);
void option_unused(list* l);
int read_option(char* s, list* options);
void option_insert(list* l, char* key, char* val);
int option_find_int(list* l, char* key, int def);
float option_find_float(list* l, char* key, float def);
char* option_find_str(list* l, char* key, char* def);
char* option_find_str_quiet(list* l, char* key, char* def);


#endif