import numpy as np

def dt2code(random_forest_tree, feature_names, class_names, func_name='f'):
    tree = random_forest_tree
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value
    n_node_samples = tree.tree_.n_node_samples

    def gen_code(left, right, threshold, features, n_node_samples, node, indent):
        output = ''
        if (threshold[node] != -2):
            output += "%sif %s <= %s:  # samples=%s\n" % (' ' * indent, features[node], str(threshold[node]), n_node_samples[node])
            if left[node] != -1:
                output += "%srules['%s'] = {'lteq', %s}\n" % (' ' * (indent+4), features[node], str(threshold[node]))
                output += gen_code(left, right, threshold, features, n_node_samples, left[node], indent+4)
            output += "%selse:\n" % (' ' * indent)
            if right[node] != -1:
                output += "%srules['%s'] = {'gteq', %s}\n" % (' ' * (indent+4), features[node], str(threshold[node]))
                output += gen_code(left, right, threshold, features, n_node_samples, right[node], indent+4)
        else:
            class_idx = np.argmax(value[node])
            output += "%sreturn %d, rules  # samples=%s\n" % (' ' * indent, class_idx, n_node_samples[node])
        return output

    code = 'def %s(%s=0):\n' % (func_name, '=0, '.join(feature_names))
    code += '    """\n'
    code += ''.join(['    %d -> %s\n' % (i, x) for (i, x) in enumerate(class_names)])
    code += '    """\n'
    code += '    rules = {}\n'
    code += gen_code(left, right, threshold, features, n_node_samples, 0, 4)
    code += "\n"
    return code

def dt2codes(random_forest_trees, feature_names, class_names, func_name='f'):
    codes = ""
    func_names = ""
    for index, r_est in enumerate(random_forest_trees.estimators_):
        current_func_name = "%s%s" % (func_name, index)
        func_names += current_func_name + ","
        codes += dt2code(r_est, feature_names, class_names, func_name=current_func_name)
    func_names = func_names[:-1]

    codes += "from collections import Counter\n"
    codes += "def predict(X):\n"
    codes += "    def majority_decision(*args):\n"
    codes += "        FUNCS = (%s)\n" % (func_names)
    codes += "        cntr = Counter()\n"
    codes += "        rules = []\n"
    codes += "        for f in FUNCS:\n"
    codes += "            predicted, rule = f(*args)\n"
    codes += "            cntr[predicted] += 1\n"
    codes += "            rules.append(rule)\n"
    codes += "        return cntr.most_common()[0][0], rules\n"
    codes += "    md_predict = []\n"
    codes += "    route_rules = []\n"
    codes += "    for md in map(lambda x: majority_decision(*x), X):\n"
    codes += "        md_predict.append(md[0])\n"
    codes += "        route_rules.append(md[1])\n"
    codes += "    return md_predict, route_rules\n"
    return codes

