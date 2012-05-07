package net.bpiwowar.qia.tasks; 
/**
 * @author B. Piwowarski <benjamin@bpiwowar.net>
 * @date 19/4/12
 */
import net.bpiwowar.kqp.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Map;
import java.util.Properties;

public class TestKQP {
  
   static public void main(String [] args) {
        int dim = 10;

        // Creating an incremental builder
        System.err.format("Creating a KEVD builder%n");
        KEVDAccumulatorDenseDouble kevd = new KEVDAccumulatorDenseDouble();

        // Add 10 vectors with $\alpha_i=1$
        System.err.format("Adding 10 vectors%n");
        for(int i = 0; i < 10; i++) {
            // Adds a random $\varphi_i$
            EigenMatrixDouble m = new EigenMatrixDouble(dim,1);
            m.randomize();
            kevd.add(new DenseDouble(m));
        }

        // Get the result $\rho \approx X Y D Y^\dagger X^\dagger$
        System.err.println("Getting the result");
        DecompositionDenseDouble d = kevd.getDecomposition();

        // --- Compute a kEVD for a subspace

        System.err.format("Creating a KEVD builder (event)%n");

        KEVDAccumulatorDenseDouble kevd_event = new KEVDAccumulatorDenseDouble();

        for(int i = 0; i < 3; i++) {
            // Adds a random $\varphi_i$
            EigenMatrixDouble m = new EigenMatrixDouble(dim,1);
            m.randomize();
            kevd_event.add(new DenseDouble(m));
        }


        // --- Compute some probabilities


        // Setup densities and events
        d = kevd.getDecomposition();
        System.err.println("Creating the density rho and event E");
        DensityDenseDouble rho = new DensityDenseDouble(kevd);
        rho.normalize();
        EventDenseDouble event = new EventDenseDouble(kevd_event);
        System.err.println("Computing some probabilities");

        // Compute the probability
        System.out.format("Probability = %g%n", rho.probability(event));

        // Conditional probability
        DensityDenseDouble rho_cond = event.project(rho).normalize();
        System.out.format("Entropy of rho/E = %g%n", rho_cond.entropy());

        // Conditional probability (orthogonal event)
        DensityDenseDouble rho_cond_orth = event.project(rho, true).normalize();
        System.out.format("Entropy of rho/not E = %g%n", rho_cond.entropy());

    }

}
