import numpy as np
from utils import rmse
def new_R(data, U, B):
    nR = np.zeros(len(data))
    for i in range(len(data)):
        nR[i] = B[:, data.newbookid.iloc[i] - 1] @ U[data.newuser_id.iloc[i] - 1, :]
    return nR

class ALS:
    def __init__(self, data, k, lamu=0.1, lamb=0.1):
        self.data = data
        self.k = k
        self.lamu = lamu
        self.lamb = lamb
        self.U = None
        self.B = None

    def fit(self):
        users = np.unique(self.data.newuser_id)
        books = np.unique(self.data.newbookid)
        nu = len(users)
        # print ("number of users: ", len(users))
        nb = len(books)

        self.U = np.ones((max(users), self.k)) / np.sqrt(self.k)
        self.B = np.ones((self.k, max(books))) / np.sqrt(self.k)

        iter = 1
        RMSE = 3
        dRMSE = 1
        rms = []
        stop = 0.0001
        max_iter = 24

        while (dRMSE > stop) and (iter < max_iter):
            for i in users:
                ind_B = self.data.newbookid[self.data.newuser_id == i] - 1
                sub_B = self.B[:, ind_B]
                nui = sub_B.shape[1]
                Ai = sub_B @ np.transpose(sub_B) + self.lamu * np.identity(self.k) * nui
                Vi = sub_B @ self.data.rating[self.data.newuser_id == i]
                self.U[i - 1, :] = np.linalg.pinv(Ai) @ Vi
            nR = new_R(self.data, self.U, self.B)
            new_RMSE = np.sqrt(np.mean((nR - self.data.rating) ** 2))
            dRMSEu = (RMSE - new_RMSE)
            RMSE = new_RMSE.copy()
      #print('dRMSE = ' + str(dRMSE))
            rms.append(RMSE)
            iter += 1
            print("step:", iter)

            for i in books:
                ind_U = self.data.newuser_id[self.data.newbookid == i] - 1
                sub_U = self.U[ind_U, :]
                nbi = sub_U.shape[0]
                Ai = np.transpose(sub_U) @ sub_U + self.lamb * np.identity(self.k) * nbi
                Vi = np.transpose(sub_U) @ self.data.rating[self.data.newbookid == i]
                self.B[:, i - 1] = np.linalg.pinv(Ai) @ Vi
            nR = new_R(self.data, self.U, self.B)
            new_RMSE = rmse(nR,self.data.rating)
            dRMSE = (RMSE - new_RMSE) #np.abs
      #dRMSE = min(dRMSEu, dRMSEb) #np.abs
            RMSE = new_RMSE.copy()
            #print('dRMSE = ' + str(dRMSE))
            #print('RMSE = ' + str(RMSE))
            print("step: ", iter)
            rms.append(RMSE)
            iter += 1
        w = {}
        w['rms'] = rms
        w['U'] = self.U
        w['B'] = self.B

        return w

    # def predict(self, user_input, book_input):
    #     U = self.U[user_input[0][0] - 1]
    #     B = self.B[:, book_input[0][0] - 1]
    #     return U @ B

    # def get_full_matrix(self, n_users, n_books):
    #     return self.U.dot(self.B)

  

    def load_weights(self, weight_path):
        # Placeholder for loading weights (implement based on storage format)
        pass

    def save_weights(self, weight_path):
        # Placeholder for saving weights (implement based on storage format)
        np.savez(weight_path, U=self.U, B=self.B)
