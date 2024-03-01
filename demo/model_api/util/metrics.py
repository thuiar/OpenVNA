import numpy as np
from sklearn.metrics import accuracy_score, f1_score, r2_score, precision_score, recall_score

__all__ = ['MSAmetrics', 'MIRmetrics']

class MSAmetrics():
    def __init__(self):
        self.metrics_dict = {
            'MOSI': self.__eval_mosi_regression,
            'MOSEI': self.__eval_mosei_regression,
            'SIMSv2': self.__eval_sims_regression
        }
    def __multiclass_acc(self, y_pred, y_true):
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def __eval_mosei_regression(self, y_pred, y_true, exclude_zero=False):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()

        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
        test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)


        mae = np.mean(np.absolute(test_preds - test_truth)).astype(np.float64)   # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a7 = self.__multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        
        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        non_zeros_binary_truth = (test_truth[non_zeros] > 0)
        non_zeros_binary_preds = (test_preds[non_zeros] > 0)

        non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
        non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average='weighted')

        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)
        acc2 = accuracy_score(binary_preds, binary_truth)
        f_score = f1_score(binary_truth, binary_preds, average='weighted')
        
        eval_results = {
            "Has0_acc_2":  round(acc2, 4),
            "Has0_F1_score": round(f_score, 4),
            "Non0_acc_2":  round(non_zeros_acc2, 4),
            "Non0_F1_score": round(non_zeros_f1_score, 4),
            "Mult_acc_5": round(mult_a5, 4),
            "Mult_acc_7": round(mult_a7, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4)
        }
        return eval_results

    def __eval_mosi_regression(self, y_pred, y_true):
        return self.__eval_mosei_regression(y_pred, y_true)
    
    def __eval_mosi_regression(self, y_pred, y_true):
        return self.__eval_mosei_regression(y_pred, y_true)
        
    def __eval_sims_regression(self, y_pred, y_true):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()
        test_preds = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth = np.clip(test_truth, a_min=-1., a_max=1.)

        # weak sentiment two classes{[-0.6, 0.0], (0.0, 0.6]}
        ms_2 = [-1.01, 0.0, 1.01]
        weak_index_l = np.where(test_truth >= -0.4)[0]
        weak_index_r = np.where(test_truth <= 0.4)[0]
        weak_index = [x for x in weak_index_l if x in weak_index_r]
        test_preds_weak = test_preds[weak_index]
        test_truth_weak = test_truth[weak_index]
        test_preds_a2_weak = test_preds_weak.copy()
        test_truth_a2_weak = test_truth_weak.copy()
        for i in range(2):
            test_preds_a2_weak[np.logical_and(test_preds_weak > ms_2[i], test_preds_weak <= ms_2[i+1])] = i
        for i in range(2):
            test_truth_a2_weak[np.logical_and(test_truth_weak > ms_2[i], test_truth_weak <= ms_2[i+1])] = i

        # two classes{[-1.0, 0.0], (0.0, 1.0]}
        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):
            test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i+1])] = i
        for i in range(2):
            test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i+1])] = i

        # three classes{[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]}
        ms_3 = [-1.01, -0.1, 0.1, 1.01]
        test_preds_a3 = test_preds.copy()
        test_truth_a3 = test_truth.copy()
        for i in range(3):
            test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i+1])] = i
        for i in range(3):
            test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i+1])] = i
        
        # five classes{[-1.0, -0.7], (-0.7, -0.1], (-0.1, 0.1], (0.1, 0.7], (0.7, 1.0]}
        ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
        test_preds_a5 = test_preds.copy()
        test_truth_a5 = test_truth.copy()
        for i in range(5):
            test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i+1])] = i
        for i in range(5):
            test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i+1])] = i
 
        mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)
        mult_a2_weak = self.__multiclass_acc(test_preds_a2_weak, test_truth_a2_weak)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        f_score = f1_score(test_truth_a2, test_preds_a2, average='weighted')
        r2 = r2_score(test_truth, test_preds)

        eval_results = {
            "Mult_acc_2_weak": mult_a2_weak,
            "Mult_acc_3": mult_a3,
            "Mult_acc_5": mult_a5,

            "Non0_acc_2": mult_a2,
            "Non0_F1_score": f_score,
            "MAE": mae,
            "Corr": corr, # Correlation Coefficient
            "R_squre":r2
        }

        return eval_results

    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName]
    
class MIRmetrics():
    
    """
    column of confusion matrix: predicted index
    row of confusion matrix: target index
    """
    def __init__(self):
        self.eval_metrics = ['acc', 'f1',  'prec', 'rec']
        self.metrics_dict = {
            'MIntRec': self.__eval_mintrec_classification,
        }

    def __eval_mintrec_classification(self, y_pred, y_true):
        y_pred = y_pred.detach().numpy()
        y_true = y_true.detach().numpy()

        y_pred = np.argmax(y_pred, axis=1)

        acc_score = self._acc_score(y_true, y_pred)
        macro_f1 = self._f1_score(y_true, y_pred)
        macro_prec = self._precision_score(y_true, y_pred)
        macro_rec = self._recall_score(y_true, y_pred)

        eval_results = {
            'acc': acc_score,
            'f1': macro_f1,
            'prec': macro_prec,
            'rec': macro_rec,
        }

        return eval_results

    def _acc_score(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)
    
    def _f1_score(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average='macro')
    
    def _precision_score(self, y_true, y_pred):
        return precision_score(y_true, y_pred, average='macro')

    def _recall_score(self, y_true, y_pred):
        return recall_score(y_true, y_pred, average='macro')
    
    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName]