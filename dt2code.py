import numpy as np

def dt2code(tree, feature_names, class_names, func_name='f'):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value
    n_node_samples = tree.tree_.n_node_samples

    def gen_code(left, right, threshold, features, n_node_samples, node, indent):
        output = ''
        if (threshold[node] != -2):
            output += "%sif %s <= %s:  # samples=%s\n" % (' ' * indent, features[node],
                                                          str(threshold[node]), n_node_samples[node])
            if left[node] != -1:
                output += gen_code(left, right, threshold, features, n_node_samples, left[node], indent+4)
            output += "%selse:\n" % (' ' * indent)
            if right[node] != -1:
                output += gen_code(left, right, threshold, features, n_node_samples, right[node], indent+4)
        else:
            class_idx = np.argmax(value[node])
            output += "%sreturn %d  # samples=%s\n" % (' ' * indent, class_idx, n_node_samples[node])
        return output

    code = 'def %s(%s=0):\n' % (func_name, '=0, '.join(feature_names))
    code += '    """\n'
    code += ''.join(['    %d -> %s\n' % (i, x) for (i, x) in enumerate(class_names)])
    code += '    """\n'
    code += gen_code(left, right, threshold, features, n_node_samples, 0, 4)
    return code

