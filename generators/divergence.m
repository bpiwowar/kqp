format short;

function dispMatrix(name, a)
  printf("Eigen::MatrixXd m%s(%d, %d);\nm%s << ", name, rows(a), columns(a), name); 

  for k=1:rows(a); 
    printf(repmat("%.30g,", 1, columns(a))(1:end-1), a(k,:)); 
    if (k != rows(a));
      printf(",");
    endif;
  endfor; 

  printf(";\n"); 
 endfunction

 function dispDiag(name, a)
  printf("Eigen::VectorXd m%s(%d);\nm%s << ", name, rows(a), name); 
  printf(repmat("%.30g,", 1, columns(a))(1:end-1), diag(a)); 
  printf(";\n"); 
 endfunction
 
function dispTest(name, description, U1, S1, U2, S2, epsilon)
  printf("// Test: %s\n", description);
  printf("namespace kqp {\n\n")
  printf("int divergence_%sTest(std::deque<std::string> &/*args*/) {\n", name);
  printf("FeatureSpace<double> fs(DenseFeatureSpace<double>::create(%d));\n", rows(U1))

  dispMatrix("U1", U1);
  dispMatrix("U2", U2);
  dispDiag("S1", S1);
  dispDiag("S2", S2);
  printf("\ndouble epsilon = %.30g;\n\n", epsilon);
  printf("Density< double > rho(fs, DenseMatrix<double>::create(mU1), Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(mU1.cols(),mU1.cols()), mS1, true);\n");
  printf("Density< double > tau(fs, DenseMatrix<double>::create(mU2), Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(mU2.cols(),mU2.cols()), mS2, true);\n");

  eid = eye(rows(U1)) / min(columns(S1) + columns(S2), rows(U1));
  
  rho = U1 * S1^2 * U1';
  tau = U2 * S2^2 * U2';
  plogp = trace(rho * logm(rho));
  plogq = trace(rho * logm((1-epsilon) * tau + epsilon * eid));
  divergence = plogp - plogq;
  printf("double divergence = rho.computeDivergence(tau, epsilon);\n");
  printf("// plogp = %.30g\n// qlogq = %.30g\n", plogp, plogq);
  printf("double expected_divergence = %.30g;\n", divergence);
  printf("KQP_LOG_INFO_F(logger, \"Divergence = %%.10g [expected %%.10g]; delta = %%.10g\", %%divergence %%expected_divergence %%(std::abs(divergence - expected_divergence)));\n");

  printf("\nreturn std::abs(divergence - expected_divergence) < 1e-10 ? 0 : 1;\n")
#  printf("\nif (std::abs(divergence - expected_divergence) < 1e-10) { String.format(\"Error while computing divergence (delta=%%g)\",abs(divergence-expected_divergence)));\n");
#  printf("\n}\n\n");
  printf("}\n");  
  printf("} // end namespace kqp\n\n");
  printf("DEFINE_TEST(\"%s\", divergence_%sTest)\n", name, name);  
endfunction

function [U,S] = decompose(rho) 
  [U, S] = eig(rho);
  idx = diag(S) > max(diag(S))*1e-16;
  S = diag(sqrt(diag(S)(idx)));
  S /= norm(S, "fro");
  U = U(:,idx);
endfunction

# --- Inits

printf("// Generated from the script divergence/divergence.m\n")
printf("#include <deque>\n")
printf("#include <kqp/kqp.hpp>\n")
printf("#include <kqp/probabilities.hpp>\n");
printf("#include <kqp/feature_matrix/dense.hpp>\n");
printf("#include \"main-tests.inc\"\n");
printf("\n\n");
printf("DEFINE_LOGGER(logger, \"kqp.test.divergence\");\n\n")


# --- First test (with epsilon = 0 and epsilon = 1e-3)

U1 = [1, 0; 0, 1]; S1 = diag([2,3]);  S1 /= norm(S1,"fro");
U2 = U1; S2 = diag([3.23, 1.234]); S2 /= norm(S2, "fro");

epsilon = 1e-3;

dispTest("simple","Simple test", U1, S1, U2, S2, 0);
dispTest("simpleEpsilon","Simple test (with epsilon)", U1, S1, U2, S2, epsilon);

# --- Second (random, bigger)

n = 10; p1 = 4; p2=6; epsilon=1e-3;
rho = rand(n, p1); rho = rho * rho'; [U1, S1] = decompose(rho);
tau = rand(n, p2); tau = tau * tau'; [U2, S2] = decompose(tau);

dispTest("full", "", U1, S1, U2, S2, epsilon);

# --- Divergence of the same density (= 0)

dispTest("zero", "", U1, S1, U1, S1, epsilon);
