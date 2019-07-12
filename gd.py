import numpy as np

class gd_group2_2d:
    
    # Initialize gd_group2_2d class attributes - i.e. the loss function and its partial derivatives
    
    def __init__(self, fn_loss, fn_grad1, fn_grad2):
        self.fn_loss = fn_loss
        self.fn_grad1 = fn_grad1
        self.fn_grad2 = fn_grad2
        
    def find_min(self, x_1_init,x_2_init, n_iter, eta, tol):
    
        x_1 = x_1_init #starting point for x1
        x_2 = x_2_init #starting point for x2
        
        loss_path = [] #empty list to record the path of the loss function
        x_1_path = [] #empty list to record the path of x1
        x_2_path = [] #empty list to record the path of x2
        
        x_1_path.append(x_1) #put the starting point into the list
        x_2_path.append(x_2) #put the starting point into the list
        loss_this = self.fn_loss(x_1,x_2) #evaluate the loss function at x1 and x2
        loss_path.append(loss_this)
        g1 = self.fn_grad1(x_1,x_2) #evaluate partial derivative wrt x1 at starting point 
        g2 = self.fn_grad2(x_1,x_2) #evaluate partial derivative wrt x2 at starting point

        for i in range(n_iter): # loop until we iterate n_iter times or until both g1,g2 < tol
            if (np.abs(g1) < tol and np.abs(g2) < tol) or np.isnan(g1) or np.isnan(g2):
                break
            g1 = self.fn_grad1(x_1,x_2) #evaluate g1 again
            g2 = self.fn_grad2(x_1,x_2) #evaluate g2 again
            x_1 += -eta * g1 #calculate new x1
            x_2 += -eta * g2 #calculate new x2
            x_1_path.append(x_1)
            x_2_path.append(x_2)
            loss_this = self.fn_loss(x_1,x_2)
            loss_path.append(loss_this)
            
        if np.isnan(g1) or np.isnan(g2):
            print('Exploded')
        elif np.abs(g1) > tol or np.abs(g2) > tol:
            print('Did not converge')
        else:
            print('Converged in {} steps.  Loss fn = {} achieved by x_1 = {} and x_2 = {}'.format(i+1, loss_this, x_1,x_2))
        
        self.loss_path = loss_path
        self.x1_path = x_1_path
        self.x2_path = x_2_path
        self.loss_fn_min = loss_this
        self.x1_at_min = x_1
        self.x2_at_min = x_2
        self.g_x1 = g1
        self.g_x2 = g2
        self.num_step = i+1
      
    
    def momentum(self, x_1_init, x_2_init, n_iter, eta, tol, alpha):
        
        x_1 = x_1_init
        x_2 = x_2_init
        
        loss_path = []
        x_1_path = []
        x_2_path = []
        
        x_1_path.append(x_1)
        x_2_path.append(x_2)
        loss_this = self.fn_loss(x_1,x_2)
        loss_path.append(loss_this)
        g1 = self.fn_grad1(x_1,x_2)
        g2 = self.fn_grad2(x_1,x_2)
        nu_1 = 0
        nu_2 = 0

        for i in range(n_iter):
            g1 = self.fn_grad1(x_1,x_2)
            g2 = self.fn_grad2(x_1,x_2)
            if (np.abs(g1) < tol and np.abs(g2) < tol) or np.isnan(g1) or np.isnan(g2):
                break

            nu_1 = alpha * nu_1 + eta * g1
            nu_2 = alpha * nu_2 + eta * g2
            x_1 += -nu_1
            x_2 += -nu_2
            x_1_path.append(x_1)
            x_2_path.append(x_2)
            loss_this = self.fn_loss(x_1,x_2)
            loss_path.append(loss_this)

        if np.isnan(g1) or np.isnan(g2):
            print('Exploded')
        elif np.abs(g1) > tol or np.abs(g2) > tol:
            print('Did not converge')
        else:
            print('Converged in {} steps.  Loss fn = {} achieved by x_1 = {} and x_2 = {}'.format(i+1, loss_this, x_1,x_2))
        
        self.loss_path = loss_path
        self.x1_path = x_1_path
        self.x2_path = x_2_path
        self.loss_fn_min_momentum = loss_this
        self.x1_at_min_momentum = x_1
        self.x2_at_min_momentum = x_2
        self.g_x1 = g1
        self.g_x2 = g2
        self.num_step = i+1
       

    def nag(self, x_1_init, x_2_init, n_iter, eta, tol, alpha):
        x_1 = x_1_init
        x_2 = x_2_init
        
        loss_path = []
        x_1_path = []
        x_2_path = []
        
        x_1_path.append(x_1)
        x_2_path.append(x_2)
        loss_this = self.fn_loss(x_1,x_2)
        loss_path.append(loss_this)
        g1 = self.fn_grad1(x_1,x_2)
        g2 = self.fn_grad2(x_1,x_2)
        nu_1 = 0
        nu_2 = 0

        for i in range(n_iter):
            # i starts from 0 so add 1
            # The formula for mu was mentioned by David Barber UCL as being Nesterovs suggestion
            mu = 1 - 3 / (i + 1 + 5) 
            g1 = self.fn_grad1(x_1 - mu*nu_1,x_2 - mu*nu_1)
            g2 = self.fn_grad2(x_2 - mu*nu_2,x_2 - mu*nu_2)
     
            if (np.abs(g1) < tol and np.abs(g2) < tol) or np.isnan(g1) or np.isnan(g2):
                break

            nu_1 = alpha * nu_1 + eta * g1
            nu_2 = alpha * nu_2 + eta * g2
            x_1 += -nu_1
            x_2 += -nu_2
            x_1_path.append(x_1)
            x_2_path.append(x_2)
            loss_this = self.fn_loss(x_1,x_2)
            loss_path.append(loss_this)

        if np.isnan(g1) or np.isnan(g2):
            print('Exploded')
        elif np.abs(g1) > tol or np.abs(g2) > tol:
            print('Did not converge')
        else:
            print('Converged in {} steps.  Loss fn = {} achieved by x_1 = {} and x_2 = {}'.format(i+1, loss_this, x_1,x_2))
        self.loss_path = loss_path
        self.x1_path = x_1_path
        self.x2_path = x_2_path
        self.loss_fn_min_nag = loss_this
        self.x1_at_min_nag = x_1
        self.x2_at_min_nag = x_2
        self.g_x1 = g1
        self.g_x2 = g2
        self.num_step = i+1
        
    

    def adam(self, x_1_init, x_2_init, n_iter, eta, tol, beta_1, beta_2):
        
        x_1 = x_1_init
        x_2 = x_2_init
        
        loss_path = []
        x_1_path = []
        x_2_path = []
        
        x_1_path.append(x_1)
        x_2_path.append(x_2)
        
        loss_this = self.fn_loss(x_1,x_2)
        loss_path.append(loss_this)
        g1 = self.fn_grad1(x_1,x_2)
        g2 = self.fn_grad2(x_1,x_2)
              
        m_1 = 0   #first moment of g1
        m_2 = 0   #first moment of g2
        
        v_1 = 0   #second moment of g1
        v_2 = 0   #second moment of g2
        
        eps = 1e-8
        
        for i in range(n_iter):       
            if (np.abs(g1) < tol and np.abs(g2) < tol) or  np.isnan(g1) or  np.isnan(g2):
                break
            
            g1 = self.fn_grad1(x_1,x_2)
            g2 = self.fn_grad2(x_1,x_2)
            
            m_1 = beta_1 * m_1 + (1 - beta_1) * g1
            m_2 = beta_1 * m_2 + (1 - beta_1) * g2
            
            v_1 = beta_2 * v_1 + (1-beta_2) * g1**2
            v_2 = beta_2 * v_2 + (1-beta_2) * g2**2
           
            m_1_hat = m_1 / (1 - beta_1**(i+1))
            m_2_hat = m_2 / (1 - beta_1**(i+1))
            
            v_1_hat = v_1 / (1 - beta_2**(i+1))
            v_2_hat = v_2 / (1 - beta_2**(i+1))
            
            x_1 -= eta * m_1_hat / (np.sqrt(v_1_hat) + eps)
            x_2 -= eta * m_2_hat / (np.sqrt(v_2_hat) + eps)
            
            x_1_path.append(x_1)
            x_2_path.append(x_2)
            loss_this = self.fn_loss(x_1,x_2)
            loss_path.append(loss_this)

        if np.isnan(g1) or np.isnan(g2):
            print('Exploded')
        elif np.abs(g1) > tol or np.abs(g2) > tol:
            print('Did not converge')
        else:
            print('Converged in {} steps.  Loss fn = {} achieved by x_1 = {} and x_2 = {}'.format(i+1, loss_this, x_1,x_2))
        self.loss_path = loss_path
        self.x1_path = x_1_path
        self.x2_path = x_2_path
        self.loss_fn_min_nag = loss_this
        self.x1_at_min_nag = x_1
        self.x2_at_min_nag = x_2
        self.g_x1 = g1
        self.g_x2 = g2
        self.num_step = i+1