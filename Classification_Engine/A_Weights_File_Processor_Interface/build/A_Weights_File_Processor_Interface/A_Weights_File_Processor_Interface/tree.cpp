#include "pch.h"
#include "tree.h"
#include "utils.h"
#include "DNLIB_Utilities.h"


float get_hierarchy_probability(float* x, tree* hier, int c)
{
    float p = 1;
    while (c >= 0) {
        p = p * x[c];
        c = hier->parent[c];
    }
    return p;
}


tree* read_tree(char* filename)
{
    tree t = { 0 };
    FILE* fp;// = fopen(filename, "r");
    fopen_s(&fp, filename, "r");

    char* line;
    int last_parent = -1;
    int group_size = 0;
    int groups = 0;
    int n = 0;
    while ((line = fgetl(fp)) != 0) {
        char* id = (char*)xcalloc(256, sizeof(char));
        int parent = -1;
        sscanf_s(line, "%s %d", id, &parent);
        t.parent = (int*)xrealloc(t.parent, (n + 1) * sizeof(int));
        t.parent[n] = parent;

        t.name = (char**)xrealloc(t.name, (n + 1) * sizeof(char*));
        t.name[n] = id;
        if (parent != last_parent) {
            ++groups;
            t.group_offset = (int*)xrealloc(t.group_offset, groups * sizeof(int));
            t.group_offset[groups - 1] = n - group_size;
            t.group_size = (int*)xrealloc(t.group_size, groups * sizeof(int));
            t.group_size[groups - 1] = group_size;
            group_size = 0;
            last_parent = parent;
        }
        t.group = (int*)xrealloc(t.group, (n + 1) * sizeof(int));
        t.group[n] = groups;
        ++n;
        ++group_size;
    }
    ++groups;
    t.group_offset = (int*)xrealloc(t.group_offset, groups * sizeof(int));
    t.group_offset[groups - 1] = n - group_size;
    t.group_size = (int*)xrealloc(t.group_size, groups * sizeof(int));
    t.group_size[groups - 1] = group_size;
    t.n = n;
    t.groups = groups;
    t.leaf = (int*)xcalloc(n, sizeof(int));
    int i;
    for (i = 0; i < n; ++i) t.leaf[i] = 1;
    for (i = 0; i < n; ++i) if (t.parent[i] >= 0) t.leaf[t.parent[i]] = 0;

    fclose(fp);
    tree* tree_ptr = (tree*)xcalloc(1, sizeof(tree));
    *tree_ptr = t;
    //error(0);
    return tree_ptr;
}