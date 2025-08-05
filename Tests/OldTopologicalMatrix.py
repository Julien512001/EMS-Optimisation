    # def create_Xsix(self, n):
    #     row = np.concatenate([np.full(2, i) if ((np.abs(i - self.params["p"]+1) % self.params["p"] == 0) and (i!=0)) else np.full(4, i) for i in range(n)], dtype=int)
    #     row = row[:n]

    #     j0 = 0
    #     i = 0
    #     col = np.zeros(n, dtype=int)
    #     while i < n:
    #         if (((row[i]+1)%(self.params["p"]) == 0) and (row[i] != 0)):
    #             col[i] = j0
    #             col[i+1] = j0 + (2*self.params["p"]-1)
    #             j0 += 1
    #             i += 2
    #         else:
    #             col[i] = j0
    #             col[i+1] = j0+1
    #             col[i+2] = j0 + (2*self.params["p"]-1)
    #             col[i+3] = (j0+1) + (2*self.params["p"]-1)
    #             j0 += 2
    #             i += 4

    #     w1 = [-1, -1, 1, 1]
    #     w3 = [-1, 1]
    #     data= []

    #     for i in range(self.params["Npsi"]):
    #         if (((i+1)%(self.params["p"]) == 0) and (i != 0)):
    #             data = np.concatenate([data, w3])
    #         else:
    #             data = np.concatenate([data, w1])

    #     Xsi_x = sp.csr_matrix((data, (row, col)))
    #     print(f'xsi.shape:{Xsi_x.shape}')
    #     return Xsi_x
    
    # def create_Xsiy(self, n):

    #     row = np.concatenate([np.full(2, i) if ((np.abs(i - self.params["p"]+1) % self.params["p"] == 0) and (i!=0)) else np.full(4, i) for i in range(n)], dtype=int)
    #     row = row[:n]

    #     j0 = 0
    #     i = 0
    #     col = np.zeros(n, dtype=int)
    #     while i < n:
    #         if (((row[i]+1)%(self.params["p"]) == 0) and (row[i] != 0)):
    #             col[i] = j0
    #             col[i+1] = j0 + (self.params["p"])
    #             j0 = j0 + self.params["p"] +1
    #             i += 2
    #         else:
    #             col[i] = j0
    #             col[i+1] = j0+1
    #             col[i+2] = j0 + self.params["p"]
    #             col[i+3] = (j0+1) + self.params["p"]
    #             j0 += 1
    #             i += 4

    #     w1 = [-1, 1, -1, 1]
    #     w3 = [-1, -1]
    #     data= []

    #     for i in range(self.params["Npsi"]):
    #         if (((i+1)%(self.params["p"]) == 0) and (i != 0)):
    #             data = np.concatenate([data, w3])
    #         else:
    #             data = np.concatenate([data, w1])


    #     Xsi_y = sp.csr_matrix((data, (row, col)))

    #     return Xsi_y
    