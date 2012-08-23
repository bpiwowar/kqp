/*
 This file is part of the Kernel Quantum Probability library (KQP).
 
 KQP is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 KQP is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with KQP.  If not, see <http://www.gnu.org/licenses/>.
 */

    // Add move semantics to MatrixBase
    Matrix(Matrix &&other) : Base() {
        Base::_check_template_params();
        this->swap(other);
    }

    Matrix &operator=(Matrix &&other) {
        this->swap(other);
        return *this;
    }

    
    template<typename OtherScalar>
    Matrix &operator=(const Eigen::Identity<OtherScalar> &other) {
        *this = Matrix::Identity(other.rows(), other.cols());
        return *this;
    }
    
    //! Debug (dump to std::cerr)
    void print() const {
        std::cerr << "=Matrix ("<<this->rows()<<" x "<<this->cols()<<")=\n" << *this << std::endl;
    }