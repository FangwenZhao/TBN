import numpy as np
import torch
def convert_to_onehot(sca_label, class_num=31):
    return np.eye(class_num)[sca_label]

class Weight:

    @staticmethod
    def cal_weight(s_label, t_label,  batch_size=32, class_num=31):
        batch_size = s_label.size()[0]


        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = convert_to_onehot(s_sca_label,class_num)          #64*10
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)       #按类别求和 1*10
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum


        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        #t_vec_label = convert_to_onehot(t_sca_label)

        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum


        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(s_sca_label)
        set_t = set(t_sca_label)
        count = 0
        for i in range(class_num):
            if i in set_s and i in set_t:
                s_tvec = s_vec_label[:, i].reshape(batch_size, -1)
                t_tvec = t_vec_label[:, i].reshape(batch_size, -1)

                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st# / np.sum(s_tvec) / np.sum(t_tvec)
                count += 1

        length = count  # len( set_s ) * len( set_t )
        if length != 0:

            weight_st = weight_st / length
        else:

            weight_st = np.array([0])
        return weight_st.astype('float32')


    @staticmethod
    def my_weight(s_labels, t_labels):
        batch_size = s_labels.size()[0]
        weights=torch.zeros((batch_size,batch_size))
        for i,s_label in enumerate(s_labels) :
            for j, t_label in enumerate(t_labels):
                weights[i,j]=s_label@t_label
        return weights/(batch_size)


    @staticmethod
    def my_weight_(s_labels, t_labels):
        batch_size = s_labels.size()[0]
        weights=torch.zeros((batch_size,batch_size))
        for i,s_label in enumerate(s_labels) :
            s_label=s_label.unsqueeze(0)
            tmp=s_label*t_labels
            weights[i]=tmp.mean(1)

        return weights/batch_size

