#include "Laplacian.hpp"

#include <Core/Index/CircularIndex.hpp>

namespace Ra {
namespace Core {
namespace Geometry {



/////////////////////
/// GLOBAL MATRIX ///
/////////////////////

// //////////////// //
// ADJACENCY MATRIX //
// //////////////// //

AdjacencyMatrix uniformAdjacency( const VectorArray< Vector3 >& p, const VectorArray< Triangle >& T ) {
    AdjacencyMatrix A( p.size(), p.size() );
    A.setZero();
    for( auto t : T ) {
        uint i = t( 0 );
        uint j = t( 1 );
        uint k = t( 2 );
        A.coeffRef( i, j ) = 1;
        A.coeffRef( j, k ) = 1;
        A.coeffRef( k, i ) = 1;
    }
    return A;
}



AdjacencyMatrix cotangentWeightAdjacency( const VectorArray< Vector3 >& p, const VectorArray< Triangle >& T ) {
    AdjacencyMatrix A( p.size(), p.size() );
    A.setZero();
    for( auto t : T ) {
        uint i = t( 0 );
        uint j = t( 1 );
        uint k = t( 2 );
        Vector3 IJ = p[j] - p[i];
        Vector3 JK = p[k] - p[j];
        Vector3 KI = p[i] - p[k];
        Scalar cotI = Vector::cotan( IJ, ( -KI ).eval() );
        Scalar cotJ = Vector::cotan( JK, ( -IJ ).eval() );
        Scalar cotK = Vector::cotan( KI, ( -JK ).eval() );
        A.coeffRef( i, j ) += cotK;
        A.coeffRef( j, i ) += cotK;
        A.coeffRef( j, k ) += cotI;
        A.coeffRef( k, j ) += cotI;
        A.coeffRef( k, i ) += cotJ;
        A.coeffRef( i, k ) += cotJ;
    }
    return ( 0.5 * A );
}



// ///////////// //
// DEGREE MATRIX //
// ///////////// //

DegreeMatrix adjacencyDegree( const AdjacencyMatrix& A ) {
    DegreeMatrix D( A.rows(), A.cols() );
    D.setZero();
    for( uint i = 0; i < D.diagonal().size(); ++i ) {
        D.coeffRef( i, i ) = A.row( i ).sum();
    }
    return D;
}



// //////////////// //
// LAPLACIAN MATRIX //
// //////////////// //

LaplacianMatrix standardLaplacian( const DegreeMatrix& D, const AdjacencyMatrix& A, const bool POSITIVE_SEMI_DEFINITE ) {
    if( POSITIVE_SEMI_DEFINITE ) {
        return ( D - A );
    }
    return ( A - D );
}



LaplacianMatrix symmetricNormalizedLaplacian( const DegreeMatrix& D, const AdjacencyMatrix& A ) {
    Sparse I( D.rows(), D.cols() );
    I.setIdentity();
    DegreeMatrix sqrt_inv_D = D.cwiseInverse().cwiseSqrt();
    return ( I - ( sqrt_inv_D * A * sqrt_inv_D ) );
}



LaplacianMatrix randomWalkNormalizedLaplacian( const DegreeMatrix& D, const AdjacencyMatrix& A ) {
    return ( D.cwiseInverse() * A );
}



LaplacianMatrix powerLaplacian( const LaplacianMatrix& L, const uint k ) {
    LaplacianMatrix lap( L.rows(), L.cols() );
    lap.setIdentity();
    for( uint i = 0; i < k; ++i ) {
        lap = L * lap;
    }
    return lap;
}



LaplacianMatrix cotangentWeightLaplacian( const VectorArray< Vector3 >& p, const VectorArray< Triangle >& T ) {
    LaplacianMatrix L( p.size(), p.size() );
    L.setZero();
    for( auto t : T ) {
        uint i = t( 0 );
        uint j = t( 1 );
        uint k = t( 2 );
        Vector3 IJ = p[j] - p[i];
        Vector3 JK = p[k] - p[j];
        Vector3 KI = p[i] - p[k];
        Scalar cotI = Vector::cotan( IJ, ( -KI ).eval() );
        Scalar cotJ = Vector::cotan( JK, ( -IJ ).eval() );
        Scalar cotK = Vector::cotan( KI, ( -JK ).eval() );
        L.coeffRef( i, j ) -= cotK;
        L.coeffRef( j, i ) -= cotK;
        L.coeffRef( j, k ) -= cotI;
        L.coeffRef( k, j ) -= cotI;
        L.coeffRef( k, i ) -= cotJ;
        L.coeffRef( i, k ) -= cotJ;
        L.coeffRef( i, i ) += cotJ + cotK;
        L.coeffRef( j, j ) += cotI + cotK;
        L.coeffRef( k, k ) += cotI + cotJ;
    }
    return ( 0.5 * L );
}



////////////////
/// ONE RING ///
////////////////

Vector3 uniformLaplacian( const Vector3& v, const VectorArray< Vector3 >& p ) {
    Vector3 L;
    L.setZero();
    for( auto pi : p ) {
        L += ( v - pi );
    }
    return L;
}



Vector3 cotangentWeightLaplacian( const Vector3& v, const VectorArray< Vector3 >& p ) {
    Vector3 L;
    L.setZero();
    uint N = p.size();
    CircularIndex i;
    i.setSize( N );
    for( uint j = 0; j < N; ++j ) {
        i.setValue( j );
        Scalar cot_a = Vector::cotan( ( v - p[i-1] ), ( p[i] - p[i-1] ) );
        Scalar cot_b = Vector::cotan( ( v - p[i+1] ), ( p[i] - p[i+1] ) );
        L += ( cot_a + cot_b ) * ( v - p[i] );
    }
    return ( 0.5 * L );
}



}
}
}