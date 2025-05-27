import streamlit as st
from naturecomm.chmm_actions import *
import numpy as np
import matplotlib.pyplot as plt
from plot import *

st.set_page_config(layout="wide")

class OnlineCHMM(CHMM):
    def __init__(self, n_clones, buffer_size=0, learning_rate=0.1, 
                 early_stopping=True, patience=50, min_delta=1e-4, *args, **kwargs):
        super().__init__(n_clones, *args, **kwargs)
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.observation_buffer = []
        self.action_buffer = []
        self.convergence_history = []

        # Early stopping parameters
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.best_likelihood = -float('inf')
        self.likelihood_history = []
        self.no_improvement_count = 0
        self.converged = False

    def add_observation(self, x_t, a_t):
        """Add observation and immediately update model with single sample"""
        if self.buffer_size > 0:
            validate_seq(x_t, a_t, self.n_clones)
            self.observation_buffer.extend(x_t)
            self.action_buffer.extend(a_t)
            if len(self.observation_buffer) > self.buffer_size:
                self.observation_buffer = self.observation_buffer[len(self.observation_buffer)-self.buffer_size:]
                self.action_buffer = self.action_buffer[len(self.observation_buffer)-self.buffer_size:]
        elif self.buffer_size == -1:
            validate_seq(x_t, a_t, self.n_clones)
            self.observation_buffer.extend(x_t)
            self.action_buffer.extend(a_t)
        else:
            if len(x_t) == 1:
                validate_seq(x_t[0], a_t[0], self.n_clones)
                self.observation_buffer = x_t  
                self.action_buffer = a_t
            else:
                validate_seq(x_t, a_t, self.n_clones)
                self.observation_buffer = x_t  
                self.action_buffer = a_t
        
        # Always update after single sample
        updated = self._update_model()
        return updated
    
    def _update_model(self):
        """Update model using current single sample"""
        x = np.array(self.observation_buffer, dtype=np.int64)
        a = np.array(self.action_buffer, dtype=np.int64)
        # Single iteration of EM for current sample
        log2_lik, mess_fwd = forward(
            self.T.transpose(0, 2, 1),
            self.Pi_x,
            self.n_clones,
            x,
            a,
            store_messages=True,
        )
        current_likelihood = log2_lik.mean()
        self.likelihood_history.append(current_likelihood)
        self.convergence_history.append(-current_likelihood)
        # Check for early stopping if enabled
        if self.early_stopping and len(self.likelihood_history) > self.patience:
            if current_likelihood > self.best_likelihood + self.min_delta:
                # Improvement found
                self.best_likelihood = current_likelihood
                self.no_improvement_count = 0
            else:
                # No significant improvement
                self.no_improvement_count += 1
                
            # Check if patience exceeded
            if self.no_improvement_count >= self.patience:
                self.converged = True
                print(f"Training converged after {len(self.likelihood_history)} updates.")
                print(f"Final log-likelihood: {current_likelihood:.4f}")
                return False
        if not self.converged:
            mess_bwd = backward(self.T, self.n_clones, x, a)
            
            C_new = np.zeros_like(self.C)
            updateC(C_new, self.T, self.n_clones, mess_fwd, mess_bwd, x, a)

            # Apply learning rate to counts
            self.C = (1 - self.learning_rate) * self.C + self.learning_rate * C_new
            
            # Update transition probabilities
            self.update_T()
            
            return True
        return False
    
def plot_metric_dashboard(monitor):
    fig, axs = plt.subplots(5, 1, figsize=(14, 18))
    plt.gcf().set_dpi(300)
    # Divergence plot
    fontsize = 17
    axs[0].plot(monitor.metrics['divergence'])
    axs[0].set_title('Policy Divergence', fontsize=fontsize)
    axs[0].set_xlim(left=1)
    axs[0].set_ylim(top=np.array(monitor.metrics['divergence'][0:]).max())
    # Entropy breakdown
    for a in range(np.array(monitor.metrics['entropy']).shape[1]):
        axs[1].plot(np.array(monitor.metrics['entropy'])[:,a], label=f'Action {a}')
    axs[1].set_title('Action Entropy', fontsize=fontsize)
    axs[1].set_xlim(left=1)

    # Volatility index
    axs[2].plot(monitor.metrics['volatility'])
    axs[2].set_title('Volatility', fontsize=fontsize)
    axs[2].set_xlim(left=1)
    # Stationary distribution
    axs[3].plot(np.array(monitor.metrics['stationary']).squeeze())
    axs[3].set_title('Stationary Dist.', fontsize=fontsize)
    axs[3].set_xlim(left=1)

    axs[4].plot(np.array(monitor.metrics['effective_clones']).squeeze())
    axs[4].set_title('Effective clones', fontsize=fontsize)
    axs[4].set_xlim(left=1)
    plt.tight_layout()
    #plt.show()
    return fig

def policy_divergence(T_old, T_new, epsilon=1e-10):
    kl1 = T_old * np.log((T_old + epsilon)/(T_new + epsilon))
    kl2 = T_new * np.log((T_new + epsilon)/(T_old + epsilon))
    return 0.5*(kl1.sum(axis=(1,2)) + kl2.sum(axis=(1,2)))

def action_policy_entropy(T):
    return -np.sum(T * np.log2(T + 1e-10), axis=(1,2))

def effective_clones(C, threshold=1e-3):
    return np.sum(np.max(C, axis=0) > threshold, axis=1)

def stationary_distribution(T):
    evals, evecs = np.linalg.eig(T.T)
    evec1 = evecs[:,np.isclose(evals, 1)]
    return evec1 / evec1.sum()

class PolicyMonitor:
    def __init__(self, n_actions, n_clones):
        self.metrics = {
            'divergence': [],
            'entropy': [],
            'volatility': [],
            'stationary': [],
            'effective_clones': []
        }
        
    def update(self, T, T_prev, C):
        self.metrics['divergence'].append(policy_divergence(T_prev, T))
        self.metrics['entropy'].append(action_policy_entropy(T))
        
        # Calculate volatility index
        if len(self.metrics['divergence']) > 10:
            recent_div = self.metrics['divergence'][-10:]
            self.metrics['volatility'].append(np.std(recent_div))
            
        # Stationary distribution analysis
        stat_dist = stationary_distribution(T.mean(axis=0))
        if len(stat_dist[0]) > 0:
            self.metrics['stationary'].append(stat_dist)

        self.metrics["effective_clones"].append(effective_clones(C))

def learn_em_T(x, a, model, n_iter=100, term_early=True, monitor= None,bar=None):
        """Run EM training, keeping E deterministic and fixed, learning T"""
        sys.stdout.flush()
        convergence = []
        pbar = trange(n_iter, position=0)
        log2_lik_old = -np.inf

        for it in pbar:
            # E
            log2_lik, mess_fwd = forward(
                model.T.transpose(0, 2, 1),
                model.Pi_x,
                model.n_clones,
                x,
                a,
                store_messages=True,
            )
            mess_bwd = backward(model.T, model.n_clones, x, a)
            updateC(model.C, model.T, model.n_clones, mess_fwd, mess_bwd, x, a)
            old_T = model.T
            # M
            model.update_T()
            if monitor != None:
                monitor.update(model.T, old_T, model.C)
            convergence.append(-log2_lik.mean())
            pbar.set_postfix(train_bps=convergence[-1])
            if log2_lik.mean() <= log2_lik_old:
                if term_early:
                    break
            log2_lik_old = log2_lik.mean()
            bar.progress(it/n_iter,text=f"Training Progress: {it}/{n_iter}")
        return monitor, convergence

st.title('cscg')

expand = st.expander("Parameters")

with expand:
    clones = st.number_input("clones", 0, 100,10)
    room_c = st.selectbox("Room", ["small_2","small_4","walls", "big"])
    learning_rate = st.number_input("learning rate",0.0,1.0,step=0.005,value=0.6)
    patience = st.number_input("Patience", 0, 1000,100)
    buffer_size_c = st.number_input("Buffer Size", -1, 10000,200)
    offset_c = st.number_input("Offset", 0, 10000,100)
    viterbi = st.checkbox("Viterbi", True)

expands = st.expander("Metrics Explanation")
f = open('metrics.md', 'r')
fileString = f.read()

with expands:
    st.markdown(
        fileString
    )

if st.button("Train"):
    if room_c == "small_4":
        room = np.array(
        [
            [1, 2, 0],
            [1, 3, 0],
            [1, 1, 0],
        ]
    )
    elif room_c == "small_2":
        room = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
        ]
    )
    elif room_c == "walls":
        H, W = 6, 8
        room = np.zeros((H, W), dtype=np.int64)
        room[:] = 0
        room[0] = 5
        room[-1] = 6
        room[:, 0] = 7
        room[:, -1] = 8
        room[0, 0] = 1
        room[0, -1] = 2
        room[-1, 0] = 3
        room[-1, -1] = 4
    else:
        room = np.array(
        [
            [1, 2, 3, 0, 3, 1, 1, 1],
            [1, 1, 3, 2, 3, 2, 3, 1],
            [1, 1, 2, 0, 1, 2, 1, 0],
            [0, 2, 1, 1, 3, 0, 0, 2],
            [3, 3, 1, 0, 1, 0, 3, 0],
            [2, 1, 2, 3, 3, 3, 2, 0],
        ]
        )

    n_emissions = room.max() + 1

    a_d, x_d, rc = datagen_structured_obs_room(room, length=100000)
    n_clones = np.ones(n_emissions, dtype=np.int64) * clones

    model = OnlineCHMM(n_clones=n_clones, learning_rate=learning_rate, early_stopping=True, patience=patience,buffer_size=buffer_size_c, min_delta=1e-5,pseudocount=2e-3,x=x_d,a=a_d)
    updates_count = 0

    monitor = PolicyMonitor(model.n_actions, n_clones)

    offset = offset_c
    bar = st.progress(0,text="Training Progress:")
    for i in trange(len(a_d)-offset):
        
        x, a = x_d[i:i+offset], a_d[i:i+offset]
        old_T = model.T
        updated = model.add_observation(x, a)

        monitor.update(model.T, old_T, model.C)
        bar.progress(i/len(a_d),text=f"Training Progress: {i}/{len(a_d)-offset}")
        if updated:
            updates_count += 1
        else:
            break
    if viterbi:
        model.learn_viterbi_T(x_d[:i+offset], a_d[:i+offset], n_iter=100,monitor=monitor)


    a, x, rc = datagen_structured_obs_room(room, length=50000)
    mon = PolicyMonitor(model.n_actions, n_clones)

    chmm = CHMM(n_clones=n_clones, pseudocount=2e-3, x=x, a=a, seed=42)  # Initialize the model
    bar.progress(0,text="Training Progress")
    mon, progression = learn_em_T(x, a, chmm,n_iter=150,monitor=mon, bar=bar)
    #mon, progression = chmm.learn_em_T(x, a, n_iter=150,monitor=mon)  # Training
    # refine learning
    chmm.pseudocount = 2e-3
    if viterbi:
        chmm.learn_viterbi_T(x, a, n_iter=100,monitor=mon)

    col1, col2 = st.columns(2)
    cmap = colors.ListedColormap(custom_colors[-n_emissions:])
    with col1:
        st.pyplot(plot_metric_dashboard(monitor))
        fig, axes = plt.subplots(figsize=(7, 4))
        axes.plot(model.convergence_history)
        axes.set_title("Loglikelihood")
        fig.tight_layout()
        st.pyplot(fig)
        graph = plot_graph(
        model, x_d, a_d, output_file="rectangular_room_graph_online.png", cmap=cmap
        )
        st.image("rectangular_room_graph_online.png")
        

    with col2:
        st.pyplot(plot_metric_dashboard(mon))
        fig, axes = plt.subplots(figsize=(7, 4))
        axes.plot(progression)
        axes.set_title("Loglikelihood")
        fig.tight_layout()
        st.pyplot(fig)
        graph = plot_graph(
        chmm, x_d, a_d, output_file="rectangular_room_graph.png", cmap=cmap
        )
        st.image("rectangular_room_graph.png")

    plt.matshow(room, cmap=cmap)
    plt.savefig("rectangular_room_layout.png")
    st.image("rectangular_room_layout.png")