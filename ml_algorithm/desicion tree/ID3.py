from math import log

class ID3DecisionTree(object):
    # calculate shannon entropy
    def calc_info_ent(self, dataset):
        num_entries = len(dataset)
        label_counts = {}
        for feat_vec in dataset:
            current_label = feat_vec[-1]
            if current_label not in label_counts.keys():
                label_counts[current_label] = 0
            label_counts[current_label] += 1
        info_ent = .0
        for key in label_counts:
            prob = float(label_counts[key]) / num_entries
            info_ent -= prob * log(prob, 2)
        return info_ent

    def create_dataset(self,):
        dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
        labels = ['no surfacing', 'flippers']
        return dataset, labels

    def split_dataset(self, dataset, axis, value):
        # TODO: dateset: ready to split
        # TODO: axis : split by feature
        # TODO: value: feature value
        ret_dataset = []
        for feat_vec in dataset:
            if feat_vec[axis] == value:
                reduced_feat_vec = feat_vec[:axis]
                reduced_feat_vec.extend(feat_vec[axis+1:])
                ret_dataset.append(reduced_feat_vec)
        return ret_dataset

    def calc_info_gain(self, dataset):
        num_features = len(dataset[0]) - 1
        info_entropy = self.calc_info_ent(dataset)
        best_info_gain = .0
        best_feature = -1
        for i in range(num_features):
            features = [example[i] for example in dataset]
            unique_vals = set(features)
            entropy = .0
            for value in unique_vals:
                sub_dataset = self.split_dataset(dataset, i, value)
                prob = len(sub_dataset) / float(len(dataset))
                entropy += prob * self.calc_info_ent(sub_dataset)
            info_gain = info_entropy - entropy
            if (info_gain > best_info_gain):
                best_info_gain = info_gain
                best_feature = i
        return best_info_gain, best_feature

def main():
    tree = ID3DecisionTree()
    dataset, labels = tree.create_dataset()
    info_gain, feature = tree.calc_info_gain(dataset)
    print(info_gain, feature)

if __name__ == '__main__':
    main()
