#include <iostream>
#include "Eigen/Eigen"
#include <iomanip>

using namespace std;
using namespace Eigen;
double Solve_PALU(const MatrixXd& A,
                  const VectorXd& b,
                  const VectorXd& x_Vero)
{
    // questa funzione mi da le matrici P, L, U
    PartialPivLU<MatrixXd> lu_decomp(A);
    VectorXd x = lu_decomp.solve(b);
    double errRel_PALU = (x_Vero - x).norm() / x_Vero.norm();

    return errRel_PALU;
}
double Solve_QR(const MatrixXd& A,
                const VectorXd& b,
                const VectorXd& x_Vero)
{
    // la funzione HouseholderQR<MatrixXd>(A) mi restituisce
    // un oggetto contenente la decomposizione QR
    // mentre la funzione solve() risolve il sistena lineare
    // tramite il metodo e fornendo in input il termine noto

    VectorXd x = HouseholderQR<MatrixXd>(A).solve(b);
    // andiamo a calcolare il valore dell'errore relativo
    double errRel_QR = (x_Vero - x).norm() / x_Vero.norm();

    return errRel_QR;
}
int main()
{
    // Write a software able to compute the linear system solution with PALU and QR
    // decomposition of the following systems:
    // adesso andiamo ad inizializzare tutte le variabili
    // che ci servono, quindi A, b, xVero
    Matrix2d A1, A2, A3;
    Vector2d b1, b2, b3, xVero;

    // inserisco i valori dati nelle matrici

    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01,-8.324762492991313e-01;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;

    // inserisco i valori dati nei vettori

    b1 << -5.169911863249772e-01,
        1.672384680188350e-01;
    b2 << -6.394645785530173e-04,
        4.259549612877223e-04;
    b3 << -6.400391328043042e-10,
        4.266924591433963e-10;

    xVero << -1.0e+0,
        -1.0e+00;

    double errRelPALU_1 = Solve_PALU(A1,b1,xVero);
    double errRelQR_1 = Solve_QR(A1,b1,xVero);
    double errRelPALU_2 = Solve_PALU(A2,b2,xVero);
    double errRelQR_2 = Solve_QR(A2,b2,xVero);
    double errRelPALU_3 = Solve_PALU(A3,b3,xVero);
    double errRelQR_3 = Solve_QR(A3,b3,xVero);

    cout << "A1: " << scientific << setprecision(15) << A1 << endl;
    cout << "Termine noto 1: " << b1 << endl;
    cout << "Errore relativo con PALU : " << errRelPALU_1 << endl;
    cout << "Errore relativo con QR : " << errRelQR_1 << endl;

    cout << "A2: " << A2 << endl;
    cout << "Termine noto 2: " << b2 << endl;
    cout << "Errore relativo con PALU : " << errRelPALU_2 << endl;
    cout << "Errore relativo con QR : " << errRelQR_2 << endl;

    cout << "A3: " << A3 << endl;
    cout << "Termine noto 3: " << b3 << endl;
    cout << "Errore relativo con PALU : " << errRelPALU_3 << endl;
    cout << "Errore relativo con QR : " << errRelQR_3 << endl;



    return 0;
}
