#include "drake/geometry/optimization/hpolyhedron.h"

#include <limits>

#include <gtest/gtest.h>

#include "drake/common/eigen_types.h"
#include "drake/common/fmt_eigen.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/maybe_pause_for_user.h"
#include "drake/common/yaml/yaml_io.h"
#include "drake/geometry/geometry_frame.h"
#include "drake/geometry/meshcat.h"
#include "drake/geometry/optimization/hyperrectangle.h"
#include "drake/geometry/optimization/point.h"
#include "drake/geometry/optimization/test_utilities.h"
#include "drake/geometry/optimization/vpolytope.h"
#include "drake/geometry/scene_graph.h"
#include "drake/geometry/test_utilities/meshcat_environment.h"
#include "drake/math/random_rotation.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/perception/point_cloud.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace geometry {
namespace optimization {

using Eigen::Matrix;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using internal::CheckAddPointInSetConstraints;
using internal::MakeSceneGraphWithShape;
using math::RigidTransformd;
using math::RotationMatrixd;
using solvers::Binding;
using solvers::Constraint;
using solvers::MathematicalProgram;
using solvers::VectorXDecisionVariable;

GTEST_TEST(HPolyhedronTest, DefaultConstructor) {
  HPolyhedron H;
  EXPECT_EQ(H.ambient_dimension(), 0);
  EXPECT_EQ(H.A().size(), 0);
  EXPECT_EQ(H.b().size(), 0);
  EXPECT_NO_THROW(H.Clone());
  EXPECT_TRUE(H.IntersectsWith(H));
  EXPECT_TRUE(H.IsBounded());
  EXPECT_FALSE(H.IsEmpty());
  EXPECT_TRUE(H.PointInSet(Eigen::VectorXd::Zero(0)));
  ASSERT_TRUE(H.MaybeGetFeasiblePoint().has_value());
  EXPECT_TRUE(H.PointInSet(H.MaybeGetFeasiblePoint().value()));
}

GTEST_TEST(HPolyhedronTest, UnitBoxTest) {
  Matrix<double, 6, 3> A;
  A << Matrix3d::Identity(), -Matrix3d::Identity();
  Vector6d b = Vector6d::Ones();

  // Test constructor.
  HPolyhedron H(A, b);
  EXPECT_EQ(H.ambient_dimension(), 3);
  EXPECT_TRUE(CompareMatrices(A, H.A()));
  EXPECT_TRUE(CompareMatrices(b, H.b()));

  // Test MakeUnitBox method.
  HPolyhedron Hbox = HPolyhedron::MakeUnitBox(3);
  EXPECT_EQ(Hbox.ambient_dimension(), 3);
  EXPECT_TRUE(CompareMatrices(A, Hbox.A()));
  EXPECT_TRUE(CompareMatrices(b, Hbox.b()));

  // Test MaybeGetPoint.
  EXPECT_FALSE(H.MaybeGetPoint().has_value());

  // Test MaybeGetFeasiblePoint.
  ASSERT_TRUE(H.MaybeGetFeasiblePoint().has_value());
  EXPECT_TRUE(H.PointInSet(H.MaybeGetFeasiblePoint().value()));

  // Test PointInSet.
  EXPECT_TRUE(H.PointInSet(Vector3d(.8, .3, -.9)));
  EXPECT_TRUE(H.PointInSet(Vector3d(-1.0, 1.0, 1.0)));
  EXPECT_FALSE(H.PointInSet(Vector3d(1.1, 1.2, 0.4)));

  // Test AddPointInSetConstraints.
  EXPECT_TRUE(CheckAddPointInSetConstraints(H, Vector3d(.8, .3, -.9)));
  EXPECT_TRUE(CheckAddPointInSetConstraints(H, Vector3d(-1.0, 1.0, 1.0)));
  EXPECT_FALSE(CheckAddPointInSetConstraints(H, Vector3d(1.1, 1.2, 0.4)));

  // Test SceneGraph constructor.
  auto [scene_graph, geom_id, context, query] =
      MakeSceneGraphWithShape(Box(2.0, 2.0, 2.0), RigidTransformd::Identity());

  HPolyhedron H_scene_graph(query, geom_id);
  EXPECT_TRUE(CompareMatrices(A, H_scene_graph.A()));
  EXPECT_TRUE(CompareMatrices(b, H_scene_graph.b()));

  ASSERT_TRUE(H_scene_graph.MaybeGetFeasiblePoint().has_value());
  EXPECT_TRUE(
      H_scene_graph.PointInSet(H_scene_graph.MaybeGetFeasiblePoint().value()));
}

GTEST_TEST(HPolyhedronTest, Move) {
  Matrix<double, 6, 3> A;
  A << Matrix3d::Identity(), -Matrix3d::Identity();
  Vector6d b = Vector6d::Ones();
  HPolyhedron orig(A, b);

  // A move-constructed HPolyhedron takes over the original data.
  HPolyhedron dut(std::move(orig));
  EXPECT_EQ(dut.ambient_dimension(), 3);
  EXPECT_TRUE(CompareMatrices(dut.A(), A));
  EXPECT_TRUE(CompareMatrices(dut.b(), b));

  // The old HPolyhedron is in a valid but unspecified state.
  EXPECT_EQ(orig.A().cols(), orig.ambient_dimension());
  EXPECT_EQ(orig.b().size(), orig.ambient_dimension());
  EXPECT_NO_THROW(orig.Clone());
}

GTEST_TEST(HPolyhedronTest, ConstructorFromVPolytope) {
  Eigen::Matrix<double, 3, 4> vert1;
  // clang-format off
  vert1 << 0, 1, 0, 0,
           0, 0, 1, 0,
           0, 0, 0, 1;
  // clang-format on
  const VPolytope vpoly1(vert1);
  const HPolyhedron hpoly1(vpoly1);
  EXPECT_EQ(hpoly1.ambient_dimension(), 3);
  EXPECT_EQ(hpoly1.A().rows(), 4);
  EXPECT_TRUE(hpoly1.PointInSet(Eigen::Vector3d(1E-3, 1E-3, 1E-3)));
  EXPECT_TRUE(hpoly1.PointInSet(Eigen::Vector3d(1 - 3E-3, 1E-3, 1E-3)));
  EXPECT_TRUE(hpoly1.PointInSet(Eigen::Vector3d(1E-3, 1 - 3E-3, 1E-3)));
  EXPECT_TRUE(hpoly1.PointInSet(Eigen::Vector3d(1E-3, 1E-3, 1 - 3E-3)));
  EXPECT_FALSE(hpoly1.PointInSet(Eigen::Vector3d(0.25, 0.25, 0.51)));
  EXPECT_FALSE(hpoly1.PointInSet(Eigen::Vector3d(-1E-5, 0.1, 0.1)));
  EXPECT_FALSE(hpoly1.PointInSet(Eigen::Vector3d(0.1, -1E-5, 0.1)));

  const Eigen::Vector3d lb(-1, -2, -3);
  const Eigen::Vector3d ub(2, 3, 4);
  const VPolytope vpoly2 = VPolytope::MakeBox(lb, ub);
  const HPolyhedron hpoly2(vpoly2);
  EXPECT_EQ(hpoly2.ambient_dimension(), 3);
  EXPECT_EQ(hpoly2.A().rows(), 6);
  EXPECT_TRUE(hpoly2.PointInSet(Eigen::Vector3d(-0.99, -1.99, -2.99)));
  EXPECT_TRUE(hpoly2.PointInSet(Eigen::Vector3d(1.99, -1.99, -2.99)));
  EXPECT_TRUE(hpoly2.PointInSet(Eigen::Vector3d(1.99, -1.99, 3.99)));
  EXPECT_FALSE(hpoly2.PointInSet(Eigen::Vector3d(0, 3.01, 0)));
  EXPECT_FALSE(hpoly2.PointInSet(Eigen::Vector3d(-1.01, 0, 0)));

  ASSERT_TRUE(hpoly1.MaybeGetFeasiblePoint().has_value());
  EXPECT_TRUE(hpoly1.PointInSet(hpoly1.MaybeGetFeasiblePoint().value()));
  ASSERT_TRUE(hpoly2.MaybeGetFeasiblePoint().has_value());
  EXPECT_TRUE(hpoly2.PointInSet(hpoly2.MaybeGetFeasiblePoint().value()));
}

GTEST_TEST(HPolyhedronTest, ConstructorFromVPolytope1D) {
  const double eps = 1e-6;

  Eigen::Matrix<double, 1, 4> vert1;
  vert1 << 1, 0, 3, 2;
  VPolytope v1(vert1);
  EXPECT_NO_THROW(HPolyhedron{v1});
  HPolyhedron h1(v1);
  EXPECT_TRUE(h1.PointInSet(Vector1d(0)));
  EXPECT_TRUE(h1.PointInSet(Vector1d(3)));
  EXPECT_FALSE(h1.PointInSet(Vector1d(0 - eps)));
  EXPECT_FALSE(h1.PointInSet(Vector1d(3 + eps)));

  Eigen::Matrix<double, 1, 1> vert2;
  vert2 << 43;
  VPolytope v2(vert2);
  HPolyhedron h2(v2);
  EXPECT_TRUE(h2.PointInSet(Vector1d(43)));
  EXPECT_FALSE(h2.PointInSet(Vector1d(43 - eps)));
  EXPECT_FALSE(h2.PointInSet(Vector1d(43 + eps)));
}

bool CheckHPolyhedronContainsVPolyhedron(const HPolyhedron& h,
                                         const VPolytope& v, double tol = 0) {
  for (int i = 0; i < v.vertices().cols(); ++i) {
    if (!h.PointInSet(v.vertices().col(i), tol)) {
      return false;
    }
  }
  return true;
}

GTEST_TEST(HPolyhedronTest, ConstructorFromVPolytopeQHullProblems) {
  // Test cases of VPolytopes that QHull cannot handle on its own.
  // Code logic in the constructor should handle these cases without any
  // QHull errors.
  const double kTol = 1e-11;

  // Case 1: Not enough points (need at least n+1 points in R^n). This
  // will throw QHull error QH6214.
  Eigen::Matrix<double, 2, 1> vert1;
  vert1 << 1, 0;
  const VPolytope vpoly1(vert1);
  EXPECT_NO_THROW(HPolyhedron{vpoly1});
  const HPolyhedron hpoly1(vpoly1);
  EXPECT_TRUE(CheckHPolyhedronContainsVPolyhedron(hpoly1, vpoly1, kTol));
  EXPECT_FALSE(hpoly1.PointInSet(Eigen::Vector2d(0, 0)));
  EXPECT_FALSE(hpoly1.PointInSet(Eigen::Vector2d(2, 0)));
  EXPECT_FALSE(hpoly1.PointInSet(Eigen::Vector2d(1, 1)));
  EXPECT_FALSE(hpoly1.PointInSet(Eigen::Vector2d(1, -1)));

  Eigen::Matrix<double, 2, 2> vert2;
  // clang-format off
  vert2 << 1, 0,
           0, 1;
  // clang-format on
  const VPolytope vpoly2(vert2);
  EXPECT_NO_THROW(HPolyhedron{vpoly2});
  const HPolyhedron hpoly2(vpoly2);
  EXPECT_TRUE(CheckHPolyhedronContainsVPolyhedron(hpoly2, vpoly2, kTol));
  EXPECT_FALSE(hpoly2.PointInSet(Eigen::Vector2d(0, 0)));
  EXPECT_FALSE(hpoly2.PointInSet(Eigen::Vector2d(1, 1)));
  EXPECT_FALSE(hpoly2.PointInSet(Eigen::Vector2d(2, -1)));
  EXPECT_FALSE(hpoly2.PointInSet(Eigen::Vector2d(-1, 2)));

  Eigen::Matrix<double, 3, 3> vert3;
  // clang-format off
  vert3 << 1, 0, 0,
           0, 1, 0,
           0, 0, 0;
  // clang-format on
  const VPolytope vpoly3(vert3);
  EXPECT_NO_THROW(HPolyhedron{vpoly3});
  const HPolyhedron hpoly3(vpoly3);
  EXPECT_TRUE(CheckHPolyhedronContainsVPolyhedron(hpoly3, vpoly3, kTol));
  EXPECT_FALSE(hpoly3.PointInSet(Eigen::Vector3d(1, 1, 0)));
  EXPECT_FALSE(hpoly3.PointInSet(Eigen::Vector3d(-1, 0, 0)));
  EXPECT_FALSE(hpoly3.PointInSet(Eigen::Vector3d(0, -1, 0)));
  EXPECT_FALSE(hpoly3.PointInSet(Eigen::Vector3d(0, 0, 1)));
  EXPECT_FALSE(hpoly3.PointInSet(Eigen::Vector3d(0, 0, -1)));

  // Case 2: VPolytope not full-dimensional (all points lie on a
  // proper affine subspace). This will throw QHull error QH6154.
  Eigen::Matrix<double, 2, 5> vert4;
  // clang-format off
  vert4 << 1, 2, 3, 4, 5,
           0, 1, 2, 3, 4;
  // clang-format on
  const VPolytope vpoly4(vert4);
  EXPECT_NO_THROW(HPolyhedron{vpoly4});
  const HPolyhedron hpoly4(vpoly4);
  EXPECT_TRUE(CheckHPolyhedronContainsVPolyhedron(hpoly4, vpoly4, kTol));

  Eigen::Matrix<double, 3, 4> vert5;
  // clang-format off
  vert5 << 0, 1, 0, 1,
           0, 0, 1, 1,
           0, 0, 0, 0;
  // clang-format on
  const VPolytope vpoly5(vert5);
  EXPECT_NO_THROW(HPolyhedron{vpoly5});
  const HPolyhedron hpoly5(vpoly5);
  EXPECT_TRUE(CheckHPolyhedronContainsVPolyhedron(hpoly5, vpoly5, kTol));

  // Case 3: VPolytope is empty
  Eigen::Matrix<double, 2, 0> vert6;
  const VPolytope vpoly6(vert6);
  EXPECT_TRUE(vpoly6.IsEmpty());
  EXPECT_EQ(vpoly6.ambient_dimension(), 2);
  EXPECT_NO_THROW(HPolyhedron{vpoly6});
  const HPolyhedron hpoly6(vpoly6);
  EXPECT_TRUE(hpoly6.IsEmpty());

  Eigen::Matrix<double, 0, 0> vert7;
  const VPolytope vpoly7(vert7);
  EXPECT_TRUE(vpoly7.IsEmpty());
  EXPECT_EQ(vpoly7.ambient_dimension(), 0);
  // Should throw an error, because HPolyhedron can't
  // handle an empty zero dimensional set.
  EXPECT_THROW(HPolyhedron{vpoly7}, std::exception);
}

bool CheckBoundedHPolyhedronAreSame(const HPolyhedron& h1,
                                    const HPolyhedron& h2, double tol = 0) {
  VPolytope v1(h1);
  VPolytope v2(h2);
  return CheckHPolyhedronContainsVPolyhedron(h1, v2, tol) &&
         CheckHPolyhedronContainsVPolyhedron(h2, v1, tol);
}

GTEST_TEST(HPolyhedronTest, ConstructorFromLinearProgramThrows) {
  // Test that a program with no variables throws.
  MathematicalProgram empty_prog;
  EXPECT_THROW(HPolyhedron{empty_prog}, std::exception);

  // Test that a program with no constraints throws.
  MathematicalProgram prog_no_constraints;
  VectorXDecisionVariable x_no_constraints =
      prog_no_constraints.NewContinuousVariables(5);
  EXPECT_THROW(HPolyhedron{prog_no_constraints}, std::exception);

  // Test that various not-linear programs throw.
  MathematicalProgram prog_socp;
  VectorXDecisionVariable x_socp = prog_socp.NewContinuousVariables(3);
  prog_socp.AddRotatedLorentzConeConstraint(MatrixXd::Identity(3, 3),
                                            VectorXd::Zero(3), x_socp);
  EXPECT_THROW(HPolyhedron{prog_socp}, std::exception);

  MathematicalProgram prog_sdp;
  solvers::MatrixXDecisionVariable x_sdp =
      prog_sdp.NewSymmetricContinuousVariables(3);
  prog_sdp.AddPositiveSemidefiniteConstraint(x_sdp);
  EXPECT_THROW(HPolyhedron{prog_sdp}, std::exception);

  MathematicalProgram prog_nlp;
  VectorXDecisionVariable x_nlp = prog_nlp.NewContinuousVariables(2);
  prog_nlp.AddConstraint(x_nlp[0] * x_nlp[1] == 1);
  EXPECT_THROW(HPolyhedron{prog_nlp}, std::exception);
}

GTEST_TEST(HPolyhedronTest, InfeasibleProgram) {
  // Test that programs with trivially infeasible lower bounds return an empty
  // HPolyhedron.
  const double kInf = std::numeric_limits<double>::infinity();
  MathematicalProgram prog1;
  VectorXDecisionVariable x1 = prog1.NewContinuousVariables(1);
  prog1.AddLinearConstraint(MatrixXd::Identity(1, 1), Vector1d(0),
                            Vector1d(-kInf), x1);
  HPolyhedron h1(prog1);
  EXPECT_EQ(h1.ambient_dimension(), 1);
  EXPECT_TRUE(h1.IsEmpty());

  MathematicalProgram prog2;
  VectorXDecisionVariable x2 = prog2.NewContinuousVariables(1);
  prog2.AddLinearConstraint(MatrixXd::Identity(1, 1), Vector1d(kInf),
                            Vector1d(0), x2);
  HPolyhedron h2(prog2);
  EXPECT_EQ(h2.ambient_dimension(), 1);
  EXPECT_TRUE(h2.IsEmpty());

  MathematicalProgram prog3;
  VectorXDecisionVariable x3 = prog3.NewContinuousVariables(4);
  prog3.AddBoundingBoxConstraint(-kInf, -kInf, x3);
  HPolyhedron h3(prog3);
  EXPECT_EQ(h3.ambient_dimension(), 4);
  EXPECT_TRUE(h3.IsEmpty());

  MathematicalProgram prog4;
  VectorXDecisionVariable x4 = prog4.NewContinuousVariables(3);
  prog4.AddBoundingBoxConstraint(kInf, kInf, x4);
  HPolyhedron h4(prog4);
  EXPECT_EQ(h4.ambient_dimension(), 3);
  EXPECT_TRUE(h4.IsEmpty());
}

GTEST_TEST(HPolyhedronTest, ConstructorFromLinearProgram) {
  const double kTol = 1e-9;

  // Make sure multiple variables, equality constraints, and bounding box
  // constraints are handled correctly.
  MathematicalProgram prog;
  VectorXDecisionVariable x1_1 = prog.NewContinuousVariables(2);
  VectorXDecisionVariable x1_2 = prog.NewContinuousVariables(1);
  prog.AddLinearEqualityConstraint(Vector2d(1, -1), 2, x1_1);
  prog.AddBoundingBoxConstraint(-5, 5, x1_1);
  prog.AddBoundingBoxConstraint(-42, 43, x1_2);
  prog.AddLinearConstraint(3 * x1_1[0] + 4 * x1_1[1] + 5 * x1_2[0] <= 6);
  HPolyhedron h1(prog);

  EXPECT_EQ(h1.ambient_dimension(), 3);
  EXPECT_EQ(h1.A().rows(), 9);
  EXPECT_EQ(h1.A().cols(), 3);
  EXPECT_EQ(h1.b().rows(), 9);
  Eigen::Matrix<double, 9, 3> A1_expected;
  VectorXd b1_expected(9);
  // clang-format off
  A1_expected <<  1, -1,  0,
                 -1,  1,  0,
                  1,  0,  0,
                 -1,  0,  0,
                  0,  1,  0,
                  0, -1,  0,
                  0,  0,  1,
                  0,  0, -1,
                  3,  4,  5;
  // clang-format on
  b1_expected << 2, -2, 5, 5, 5, 5, 43, 42, 6;
  HPolyhedron h1_expected(A1_expected, b1_expected);
  EXPECT_TRUE(CheckBoundedHPolyhedronAreSame(h1, h1_expected, kTol));

  // Check that PointInSet constraints work with HPolyhedron.
  HPolyhedron h2_expected(h1_expected);
  MathematicalProgram prog2;
  VectorXDecisionVariable x2 =
      prog2.NewContinuousVariables(h2_expected.ambient_dimension());
  h2_expected.AddPointInSetConstraints(&prog2, x2);
  HPolyhedron h2(prog2);
  EXPECT_TRUE(CheckBoundedHPolyhedronAreSame(h2, h2_expected, kTol));

  // Check that PointInSet constraints work with VPolytope.
  Eigen::Matrix<double, 1, 2> vpoly_points;
  vpoly_points << -1, 1;
  VPolytope vpoly(vpoly_points);
  MathematicalProgram prog3;
  VectorXDecisionVariable x3 = prog3.NewContinuousVariables(1);
  vpoly.AddPointInSetConstraints(&prog3, x3);
  HPolyhedron h3(prog3);

  // One variable for each point.
  EXPECT_EQ(h3.ambient_dimension(), 3);
  // 2 rows for the bounding box on each weight variable, 2 rows for the
  // equality constraint on x3, 2 rows for the equality constraint summing the
  // weights to one.
  EXPECT_EQ(h3.A().rows(), 8);
  EXPECT_EQ(h3.A().cols(), 3);
  EXPECT_EQ(h3.b().rows(), 8);
  Eigen::Matrix<double, 8, 3> A3_expected;
  VectorXd b3_expected(8);
  // clang-format off
  A3_expected <<  0,  1,  0,  // α₁ <= 1
                  0, -1,  0,  // α₁ >= 0
                  0,  0,  1,  // α₂ <= 1
                  0,  0, -1,  // α₂ >= 0
                 -1, -1,  1,  // -x + (-1)α₁ + (1)α₂ <= 0
                  1,  1, -1,  // -x + (-1)α₁ + (1)α₂ >= 0
                  0,  1,  1,  // α₁ + α₂ <= 1
                  0, -1, -1;  // α₁ + α₂ >= 1
  // clang-format on
  b3_expected << 1, 0, 1, 0, 0, 0, 1, -1;
  HPolyhedron h3_expected(A3_expected, b3_expected);
  EXPECT_TRUE(CheckBoundedHPolyhedronAreSame(h3, h3_expected, kTol));

  // Check that PointInSet constraints work with Point.
  Point point(Eigen::Vector2d(1, 2));
  MathematicalProgram prog4;
  VectorXDecisionVariable x4 = prog4.NewContinuousVariables(2);
  point.AddPointInSetConstraints(&prog4, x4);
  HPolyhedron h4(prog4);
  EXPECT_EQ(h4.ambient_dimension(), 2);
  EXPECT_EQ(h4.A().rows(), 4);
  EXPECT_EQ(h4.A().cols(), 2);
  EXPECT_EQ(h4.b().rows(), 4);
  Eigen::Matrix<double, 4, 2> A4_expected;
  // clang-format off
  A4_expected <<  1,  0,
                 -1,  0,
                  0,  1,
                  0, -1;
  // clang-format on
  VectorXd b4_expected(4);
  b4_expected << 1, -1, 2, -2;
  HPolyhedron h4_expected(A4_expected, b4_expected);
  EXPECT_TRUE(CheckBoundedHPolyhedronAreSame(h4, h4_expected, kTol));

  // Check that it works for unbounded HPolyhedra.
  MathematicalProgram prog5;
  VectorXDecisionVariable x5 = prog5.NewContinuousVariables(1);
  prog5.AddLinearConstraint(x5[0] <= 0);
  HPolyhedron h5(prog5);
  EXPECT_EQ(h5.ambient_dimension(), 1);
  EXPECT_EQ(h5.A().rows(), 1);
  EXPECT_EQ(h5.A().cols(), 1);
  EXPECT_EQ(h5.b().rows(), 1);
  EXPECT_EQ(h5.A()(0, 0), 1);
  EXPECT_EQ(h5.b()(0), 0);
}

GTEST_TEST(HPolyhedronTest, L1BallTest) {
  Matrix<double, 8, 3> A;
  VectorXd b = VectorXd::Ones(8);
  // clang-format off
  A <<   1,  1,  1,
        -1,  1,  1,
         1, -1,  1,
        -1, -1,  1,
         1,  1, -1,
        -1,  1, -1,
         1, -1, -1,
        -1, -1, -1;
  // clang-format on

  // Test MakeL1Ball method.
  HPolyhedron H_L1_box = HPolyhedron::MakeL1Ball(3);
  EXPECT_EQ(H_L1_box.ambient_dimension(), 3);
  EXPECT_TRUE(CompareMatrices(A, H_L1_box.A()));
  EXPECT_TRUE(CompareMatrices(b, H_L1_box.b()));

  ASSERT_TRUE(H_L1_box.MaybeGetFeasiblePoint().has_value());
  EXPECT_TRUE(H_L1_box.PointInSet(H_L1_box.MaybeGetFeasiblePoint().value()));
}

GTEST_TEST(HPolyhedronTest, ArbitraryBoxTest) {
  RigidTransformd X_WG(RotationMatrixd::MakeZRotation(M_PI / 2.0),
                       Vector3d(-4.0, -5.0, -6.0));
  auto [scene_graph, geom_id, context, query] =
      MakeSceneGraphWithShape(Box(1.0, 2.0, 3.0), X_WG);
  HPolyhedron H(query, geom_id);

  EXPECT_EQ(H.ambient_dimension(), 3);
  // Rotated box should end up with lb=[-5,-5.5,-7.5], ub=[-3,-4.5,-4.5].
  Vector3d in1_W{-4.9, -5.4, -7.4}, in2_W{-3.1, -4.6, -4.6},
      out1_W{-5.1, -5.6, -7.6}, out2_W{-2.9, -4.4, -4.4};

  EXPECT_LE(query.ComputeSignedDistanceToPoint(in1_W)[0].distance, 0.0);
  EXPECT_LE(query.ComputeSignedDistanceToPoint(in2_W)[0].distance, 0.0);
  EXPECT_GE(query.ComputeSignedDistanceToPoint(out1_W)[0].distance, 0.0);
  EXPECT_GE(query.ComputeSignedDistanceToPoint(out2_W)[0].distance, 0.0);

  EXPECT_TRUE(H.PointInSet(in1_W));
  EXPECT_TRUE(H.PointInSet(in2_W));
  EXPECT_FALSE(H.PointInSet(out1_W));
  EXPECT_FALSE(H.PointInSet(out2_W));

  ASSERT_TRUE(H.MaybeGetFeasiblePoint().has_value());
  EXPECT_TRUE(H.PointInSet(H.MaybeGetFeasiblePoint().value()));

  EXPECT_TRUE(CheckAddPointInSetConstraints(H, in1_W));
  EXPECT_TRUE(CheckAddPointInSetConstraints(H, in2_W));
  EXPECT_FALSE(CheckAddPointInSetConstraints(H, out1_W));
  EXPECT_FALSE(CheckAddPointInSetConstraints(H, out2_W));

  // Test reference_frame frame.
  SourceId source_id = scene_graph->RegisterSource("F");
  FrameId frame_id = scene_graph->RegisterFrame(source_id, GeometryFrame("F"));
  auto context2 = scene_graph->CreateDefaultContext();
  const RigidTransformd X_WF{math::RollPitchYawd(.1, .2, 3),
                             Vector3d{.5, .87, .1}};
  const FramePoseVector<double> pose_vector{{frame_id, X_WF}};
  scene_graph->get_source_pose_port(source_id).FixValue(context2.get(),
                                                        pose_vector);
  auto query2 =
      scene_graph->get_query_output_port().Eval<QueryObject<double>>(*context2);
  HPolyhedron H_F(query2, geom_id, frame_id);

  const RigidTransformd X_FW = X_WF.inverse();
  EXPECT_TRUE(H_F.PointInSet(X_FW * in1_W));
  EXPECT_TRUE(H_F.PointInSet(X_FW * in2_W));
  EXPECT_FALSE(H_F.PointInSet(X_FW * out1_W));
  EXPECT_FALSE(H_F.PointInSet(X_FW * out2_W));

  const double kTol = 1e-14;
  ASSERT_TRUE(H_F.MaybeGetFeasiblePoint().has_value());
  EXPECT_TRUE(H_F.PointInSet(H_F.MaybeGetFeasiblePoint().value(), kTol));
}

GTEST_TEST(HPolyhedronTest, HalfSpaceTest) {
  RigidTransformd X_WG(RotationMatrixd::MakeYRotation(M_PI / 2.0),
                       Vector3d(-1.2, -2.1, -6.4));
  auto [scene_graph, geom_id, context, query] =
      MakeSceneGraphWithShape(HalfSpace(), X_WG);
  HPolyhedron H(query, geom_id);

  EXPECT_EQ(H.ambient_dimension(), 3);

  // Rotated HalfSpace should be x <= -1.2.
  Vector3d in1_W{-1.21, 0.0, 0.0}, in2_W{-1.21, 2., 3.}, out1_W{-1.19, 0, 0},
      out2_W{-1.19, 2., 3.};

  EXPECT_LE(query.ComputeSignedDistanceToPoint(in1_W)[0].distance, 0.0);
  EXPECT_LE(query.ComputeSignedDistanceToPoint(in2_W)[0].distance, 0.0);
  EXPECT_GE(query.ComputeSignedDistanceToPoint(out1_W)[0].distance, 0.0);
  EXPECT_GE(query.ComputeSignedDistanceToPoint(out2_W)[0].distance, 0.0);

  EXPECT_TRUE(H.PointInSet(in1_W));
  EXPECT_TRUE(H.PointInSet(in2_W));
  EXPECT_FALSE(H.PointInSet(out1_W));
  EXPECT_FALSE(H.PointInSet(out2_W));

  ASSERT_TRUE(H.MaybeGetFeasiblePoint().has_value());
  EXPECT_TRUE(H.PointInSet(H.MaybeGetFeasiblePoint().value()));
}

GTEST_TEST(HPolyhedronTest, UnitBox6DTest) {
  HPolyhedron H = HPolyhedron::MakeUnitBox(6);
  EXPECT_EQ(H.ambient_dimension(), 6);

  Vector6d in1_W{Vector6d::Constant(-.99)}, in2_W{Vector6d::Constant(.99)},
      out1_W{Vector6d::Constant(-1.01)}, out2_W{Vector6d::Constant(1.01)};

  EXPECT_TRUE(H.PointInSet(in1_W));
  EXPECT_TRUE(H.PointInSet(in2_W));
  EXPECT_FALSE(H.PointInSet(out1_W));
  EXPECT_FALSE(H.PointInSet(out2_W));

  ASSERT_TRUE(H.MaybeGetFeasiblePoint().has_value());
  EXPECT_TRUE(H.PointInSet(H.MaybeGetFeasiblePoint().value()));
}

GTEST_TEST(HPolyhedronTest, InscribedEllipsoidTest) {
  // Test a unit box.
  HPolyhedron H = HPolyhedron::MakeUnitBox(3);
  Hyperellipsoid E = H.MaximumVolumeInscribedEllipsoid();
  // The exact tolerance will be solver dependent; this is (hopefully)
  // conservative enough.
  const double kTol = 1e-4;
  EXPECT_TRUE(CompareMatrices(E.center(), Vector3d::Zero(), kTol));
  EXPECT_TRUE(CompareMatrices(E.A().transpose() * E.A(),
                              Matrix3d::Identity(3, 3), kTol));

  // A non-trivial example, taken some real problem data.  The addition of the
  // extra half-plane constraints cause the optimal ellipsoid to be far from
  // axis-aligned.
  Matrix<double, 8, 3> A;
  Matrix<double, 8, 1> b;
  // clang-format off
  A << Matrix3d::Identity(),
       -Matrix3d::Identity(),
       .9, -.3, .1,
       .9, -.3, .1;
  b << 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 1.3, 0.8;
  // clang-format on
  HPolyhedron H2(A, b);
  Hyperellipsoid E2 = H2.MaximumVolumeInscribedEllipsoid();
  // Check that points just inside the boundary of the ellipsoid are inside the
  // polytope.
  Matrix3d C = E2.A().inverse();
  RandomGenerator generator;
  for (int i = 0; i < 10; ++i) {
    const RotationMatrixd R = math::UniformlyRandomRotationMatrix(&generator);
    SCOPED_TRACE(
        fmt::format("With random rotation matrix\n{}", fmt_eigen(R.matrix())));
    Vector3d x = C * R.matrix() * Vector3d(0.99, 0.0, 0.0) + E2.center();
    EXPECT_TRUE(E2.PointInSet(x));
    EXPECT_TRUE(H2.PointInSet(x));
  }

  // Make sure the ellipsoid touches the polytope, by checking that the minimum
  // residual, bᵢ − aᵢd − |aᵢC|₂, is zero.
  const VectorXd polytope_halfspace_residue =
      b - A * E2.center() - ((A * C).rowwise().lpNorm<2>());
  EXPECT_NEAR(polytope_halfspace_residue.minCoeff(), 0, kTol);

  // Check numerical stability for poorly-formed A and b matrices
  MatrixXd A2(24, 12);
  VectorXd b2(24);
  // clang-format off
  A2 << MatrixXd::Identity(12, 12),
        -MatrixXd::Identity(12, 12);
  b2 << VectorXd::Ones(12),
        VectorXd::Zero(12);
  // clang-format on
  for (int i = 0; i < A2.rows(); ++i) {
    double scaling_factor = std::pow(10, i - 12);
    A2.row(i) *= scaling_factor;
    b2(i) *= scaling_factor;
  }
  HPolyhedron H4(A2, b2);

  // Check that we can compute the maximum volume inscribed ellipsoid of the
  // HPolyhedron defined by the ill-formed matrix.
  EXPECT_NO_THROW(unused(H4.MaximumVolumeInscribedEllipsoid()));

  // An example for which the inscribed ellipsoid has very small volume.
  // The Clarabel solver can fail with small volume polyhedra.
  Matrix<double, 33, 10> A3;
  Matrix<double, 33, 1> b3;

  A3 << 0.773199, 0.11942, 0.128259, -0.576654, -0.273927, 0.0783283, 0, 0, 0,
      0, 0.186593, 0.0084722, 0.0647332, 0.219393, -0.973221, 0.169623, 0, 0, 0,
      0, -1.72e-06, -1.12e-06, -0.705771, -4.12e-06, 0.0561511, 0.705765, 0, 0,
      0, 0, -0.0007536, 0.344005, 0.0548817, -0.452125, -0.748107, 0.0554738, 0,
      0, 0, 0, 0.583244, -0.121236, -0.136299, -0.761597, 0.287719, -0.0832273,
      0, 0, 0, 0, -0.708719, -0.00191545, -0.00212362, 0.705434, 0.00923656,
      -0.00198491, 0, 0, 0, 0, -0.501993, 0.412856, -0.338715, -0.70477,
      0.107421, -0.0589722, 0, 0, 0, 0, -0.0079596, -0.762563, -0.593568,
      -0.340412, 0.33152, -0.287148, 0, 0, 0, 0, 0.134088, 0.123581, -0.825216,
      -0.373107, 0.302132, -0.4533, 0, 0, 0, 0, 0.213252, -0.737729, -0.616836,
      -0.113148, 0.322648, -0.305637, 0, 0, 0, 0, 0.340497, -0.758502,
      -0.0636514, 0.540365, 0.094898, -0.0150524, 0, 0, 0, 0, -0.170974,
      -0.0768362, -0.0506272, -0.0955827, 0.975901, -0.148159, 0, 0, 0, 0,
      -0.21628, 0.251743, -0.477133, 0.288735, -0.0411648, -0.74962, 0, 0, 0, 0,
      0.55041, 0.546512, -0.137804, 0.64413, -0.0659659, -0.0284664, 0, 0, 0, 0,
      0.0732983, 0.319178, -0.452745, 0.412174, -0.0584214, -0.710916, 0, 0, 0,
      0, 0.313665, 0.38977, 0.507381, 0.15648, 0.0293681, -0.677222, 0, 0, 0, 0,
      0.192168, 0.0863609, 0.78292, 0.107431, 0.0577422, -0.559493, 0, 0, 0, 0,
      -0.293778, -0.405425, 0.597923, 0.126046, 0.0514021, 0.599926, 0, 0, 0, 0,
      0.723097, 0.413477, 0.590019, 0.222499, -0.298303, 0.254303, 0, 0, 0, 0,
      0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0.558002, 0.382054, -0.628086,
      -0.0287501, 0.07297, -0.376861, 0, 0, 0, 0, -0.942613, -0.315724,
      -0.0323777, 0.0963124, 0.0381817, 0.00411583, 0, 0, 0, 0, -0.184564,
      0.090047, -0.192762, -0.321208, 0.343848, -0.83622, 0, 0, 0, 0, -0.102123,
      0.00305548, -0.590316, -0.444868, -0.569769, 0.344303, 0, 0, 0, 0,
      -0.0224168, -0.219739, 0.926411, -0.302016, -0.0419215, -0.00183036, 0, 0,
      0, 0, 0.846166, 0.168459, 0.152309, 0.0430452, -0.0841153, 0.472756, 0, 0,
      0, 0, -0.421626, -0.350029, 0.830406, 0.0106941, -0.0990932, -0.0142503,
      0, 0, 0, 0, -0.0812237, 0.325186, -0.125941, 0.025877, -0.886702,
      -0.291353, 0, 0, 0, 0, -0.183157, 0.0605173, -0.405228, -0.395436,
      0.599727, 0.531544, 0, 0, 0, 0, 0.352718, -0.107803, 0.404796, 0.0921706,
      0.662548, -0.502636, 0, 0, 0, 0, -0.00407484, 0.232374, 0.227578,
      0.812461, -0.372462, -0.308825, 0, 0, 0, 0, -0.148314, 0.192827,
      0.0480374, 0.935771, -0.159495, 0.193411, 0, 0, 0, 0, -0.285996, 0.805894,
      0.194024, 0.30219, 0.2991, 0.224314,

      b3 << 3.78472, 0.426809, -0.771432, 0.731582, 1.06898, -0.874681,
      -0.548086, -6.56011, -0.127327, -4.91151, -1.46553, 0.5079, 1.19766,
      9.40101, 3.30244, 6.3117, 4.31042, -1.11083, 8.62258, 1.71977, 0.0343364,
      -2.4909, -3.66465, -5.66561, -0.0176519, 5.70697, -1.03391, -7.76507,
      5.76258, 4.85508, 0.443183, 4.70376, 8.2951;

  Eigen::Matrix<double, 10, 1> desired_center;
  Eigen::Matrix<double, 10, 10> desired_C;

  desired_center << 3.51591, 9.25533, 3.3651, 2.29865, 2.57861, 2.06692,
      1.97387, 3.20778, 8.46634, 5.83027;

  desired_C << 17297.2, 10139.9, 6699.42, 4365.81, -9540.36, 5882.58, 285.673,
      -484.43, -225.836, 5.82859, 10139.9, 8552.21, 2015.31, 1687.37, -4107.56,
      1328.02, 75.6002, -457.071, -120.824, -55.4118, 6699.42, 2015.31, 26245.5,
      3846.13, -2719.65, -3775.5, 86.464, -569.165, -196.875, -59.9433, 4365.81,
      1687.37, 3846.13, 3478.35, -5571.91, 4419.85, 273.61, -441.17, -171.417,
      7.98864, -9540.36, -4107.56, -2719.65, -5571.91, 102708, 5600.47, 1485.8,
      -4383.44, -1707.49, -208.072, 5882.58, 1328.02, -3775.5, 4419.85, 5600.47,
      33336, 668.665, -3328.38, -1263.94, -246.303, 285.673, 75.6002, 86.464,
      273.61, 1485.8, 668.665, 8747.95, -1015.75, 989.829, -556.727, -484.43,
      -457.071, -569.165, -441.17, -4383.44, -3328.38, -1015.75, 5465.82,
      2415.72, 28.5514, -225.836, -120.824, -196.875, -171.417, -1707.49,
      -1263.94, 989.829, 2415.72, 3341.51, -1969.43, 5.82859, -55.4118,
      -59.9433, 7.98864, -208.072, -246.303, -556.727, 28.5514, -1969.43,
      3392.38;

  HPolyhedron H_small(A3, b3);
  EXPECT_TRUE(H_small.IsBounded());
  EXPECT_FALSE(H_small.IsEmpty());
  Hyperellipsoid E_small = H_small.MaximumVolumeInscribedEllipsoid();

  const double kTolCenter = 1e-4;
  const double kTolEMat = 1e-6;

  EXPECT_TRUE(CompareMatrices(E_small.center(), desired_center, kTolCenter));
  EXPECT_TRUE(
      CompareMatrices(E_small.A().inverse(), desired_C.inverse(), kTolEMat));
}

GTEST_TEST(HPolyhedronTest, ChebyshevCenter) {
  HPolyhedron box = HPolyhedron::MakeUnitBox(6);
  EXPECT_TRUE(CompareMatrices(box.ChebyshevCenter(), Vector6d::Zero(), 1e-6));
}

// A rotated long thin rectangle in 2 dimensions.
GTEST_TEST(HPolyhedronTest, ChebyshevCenter2) {
  Matrix<double, 4, 2> A;
  Vector4d b;
  // clang-format off
  A << -2, -1,  // 2x + y ≥ 4
        2,  1,  // 2x + y ≤ 6
       -1,  2,  // x - 2y ≥ 2
        1, -2;  // x - 2y ≤ 8
  b << -4, 6, -2, 8;
  // clang-format on
  HPolyhedron H(A, b);
  const VectorXd center = H.ChebyshevCenter();
  EXPECT_TRUE(H.PointInSet(center));
  // For the rectangle, the center should have distance = 1.0 from the first
  // two half-planes, and ≥ 1.0 for the other two.
  const VectorXd distance = b - A * center;
  EXPECT_NEAR(distance[0], 1.0, 1e-6);
  EXPECT_NEAR(distance[1], 1.0, 1e-6);
  EXPECT_GE(distance[2], 1.0 - 1e-6);
  EXPECT_GE(distance[3], 1.0 - 1e-6);
}

GTEST_TEST(HpolyhedronTest, Scale) {
  const double kTol = 1e-12;
  const HPolyhedron H = HPolyhedron::MakeUnitBox(3);
  const double kScale = 2.0;

  // The original volume is 2x2x2 = 8.
  // The new volume should be 16.
  HPolyhedron H_scaled = H.Scale(kScale);
  VPolytope V(H_scaled);
  EXPECT_NEAR(V.CalcVolume(), 16.0, kTol);
  // The vertices should be pow(16,1/3)/2.
  const double kVertexValue = std::pow(16.0, 1.0 / 3.0) / 2.0;
  for (int i = 0; i < V.vertices().rows(); ++i) {
    for (int j = 0; j < V.vertices().cols(); ++j) {
      EXPECT_NEAR(std::abs(V.vertices()(i, j)), kVertexValue, kTol);
    }
  }

  // Again with the center specified explicitly.
  H_scaled = H.Scale(kScale, Vector3d::Zero());
  V = VPolytope(H_scaled);
  EXPECT_NEAR(V.CalcVolume(), 16.0, kTol);

  // Again with the center in the bottom corner.
  H_scaled = H.Scale(1.0 / 8.0, Vector3d::Constant(-1.0));
  V = VPolytope(H_scaled);
  EXPECT_NEAR(V.CalcVolume(), 1.0, kTol);
  EXPECT_TRUE(H_scaled.PointInSet(Vector3d::Constant(-0.01)));
  EXPECT_FALSE(H_scaled.PointInSet(Vector3d::Constant(0.01)));
  EXPECT_TRUE(H_scaled.PointInSet(Vector3d::Constant(-0.99)));
  EXPECT_FALSE(H_scaled.PointInSet(Vector3d::Constant(-1.01)));

  ASSERT_TRUE(H_scaled.MaybeGetFeasiblePoint().has_value());
  EXPECT_TRUE(H_scaled.PointInSet(H_scaled.MaybeGetFeasiblePoint().value()));

  // Shrink to a point.
  const Vector3d kPoint = Vector3d::Constant(-1.0);
  H_scaled = H.Scale(0, kPoint);
  // A*point == b.
  EXPECT_TRUE(CompareMatrices(H_scaled.A() * kPoint, H_scaled.b(), kTol));
}

// Scale supports unbounded sets.
GTEST_TEST(HPolyhedronTest, Scale2) {
  const double kTol = 1e-14;
  // The ice cream cone in 2d. y>=x, y>=-x.
  Eigen::Matrix2d A;
  A << 1, -1, 1, 1;
  HPolyhedron H(A, Vector2d::Zero());

  const double kScale = 0.25;
  // Scaling about the origin should have no effect.
  HPolyhedron H_scaled = H.Scale(kScale, Vector2d::Zero());
  EXPECT_TRUE(CompareMatrices(H.A(), H_scaled.A(), kTol));
  EXPECT_TRUE(CompareMatrices(H.b(), H_scaled.b(), kTol));

  // Scaling about the point (0,1) will move the cone up, to
  // y >= x + 0.5, y >= -x - 0.5.
  H_scaled = H.Scale(kScale, Vector2d{0, 1});
  EXPECT_TRUE(CompareMatrices(H_scaled.A(), H.A(), kTol));
  EXPECT_TRUE(CompareMatrices(H_scaled.b(), Vector2d{-0.5, 0.5}, kTol));
}

// The original set has no volume.
GTEST_TEST(HPolyhedronTest, Scale3) {
  // Make a square in the xz plane, with y=0.
  Eigen::MatrixXd A(6, 3);
  // clang-format off
  A <<  1,  0,  0,  // x <= 1
       -1,  0,  0,  // x >= -1
        0,  1,  0,  // y <= 0
        0, -1,  0,  // y >= 0
        0,  0,  1,  // z <= 1
        0,  0, -1;  // z >= -1
  // clang-format on
  VectorXd b(6);
  b << 1, 1, 0, 0, 1, 1;
  HPolyhedron H(A, b);

  const double kScale = 2.0;
  const double kOffset = 1e-6;
  HPolyhedron H_scaled = H.Scale(kScale, Vector3d::Zero());
  const double kVertexValue = std::pow(16.0, 1.0 / 3.0) / 2.0;
  EXPECT_TRUE(H_scaled.PointInSet(
      Vector3d{kVertexValue - kOffset, 0, kVertexValue - kOffset}));
  EXPECT_FALSE(H_scaled.PointInSet(
      Vector3d{kVertexValue + kOffset, 0, kVertexValue + kOffset}));

  // center does not need to be in the set.
  H_scaled = H.Scale(kScale, Vector3d{2, 0, 0});
  EXPECT_FALSE(H_scaled.PointInSet(Vector3d{1, 0, 0}));
  EXPECT_TRUE(H_scaled.PointInSet(Vector3d{-1, 0, 0}));

  ASSERT_TRUE(H_scaled.MaybeGetFeasiblePoint().has_value());
  EXPECT_TRUE(H_scaled.PointInSet(H_scaled.MaybeGetFeasiblePoint().value()));
}

GTEST_TEST(HPolyhedronTest, CloneTest) {
  HPolyhedron H = HPolyhedron::MakeBox(Vector3d{-3, -4, -5}, Vector3d{6, 7, 8});
  std::unique_ptr<ConvexSet> clone = H.Clone();
  EXPECT_EQ(clone->ambient_dimension(), H.ambient_dimension());
  HPolyhedron* pointer = dynamic_cast<HPolyhedron*>(clone.get());
  ASSERT_NE(pointer, nullptr);
  EXPECT_TRUE(CompareMatrices(H.A(), pointer->A()));
  EXPECT_TRUE(CompareMatrices(H.b(), pointer->b()));
}

GTEST_TEST(HPolyhedronTest, NonnegativeScalingTest) {
  const Vector3d lb{1, 1, 1}, ub{2, 3, 4};
  HPolyhedron H = HPolyhedron::MakeBox(lb, ub);

  MathematicalProgram prog;
  auto x = prog.NewContinuousVariables(3, "x");
  auto t = prog.NewContinuousVariables(1, "t")[0];

  std::vector<Binding<Constraint>> constraints =
      H.AddPointInNonnegativeScalingConstraints(&prog, x, t);

  EXPECT_EQ(constraints.size(), 2);

  const double tol = 0;

  prog.SetInitialGuess(x, 0.99 * ub);
  prog.SetInitialGuess(t, 1.0);
  EXPECT_TRUE(prog.CheckSatisfiedAtInitialGuess(constraints, tol));

  prog.SetInitialGuess(x, 1.01 * ub);
  prog.SetInitialGuess(t, 1.0);
  EXPECT_FALSE(prog.CheckSatisfiedAtInitialGuess(constraints, tol));

  prog.SetInitialGuess(x, 0.99 * ub);
  prog.SetInitialGuess(t, -0.01);
  EXPECT_FALSE(prog.CheckSatisfiedAtInitialGuess(constraints, tol));

  prog.SetInitialGuess(x, 0.49 * ub);
  prog.SetInitialGuess(t, 0.5);
  EXPECT_TRUE(prog.CheckSatisfiedAtInitialGuess(constraints, tol));

  prog.SetInitialGuess(x, 0.51 * ub);
  prog.SetInitialGuess(t, 0.5);
  EXPECT_FALSE(prog.CheckSatisfiedAtInitialGuess(constraints, tol));

  prog.SetInitialGuess(x, 1.99 * ub);
  prog.SetInitialGuess(t, 2.0);
  EXPECT_TRUE(prog.CheckSatisfiedAtInitialGuess(constraints, tol));

  prog.SetInitialGuess(x, 2.01 * ub);
  prog.SetInitialGuess(t, 2.0);
  EXPECT_FALSE(prog.CheckSatisfiedAtInitialGuess(constraints, tol));
}

bool PointInScaledSet(const solvers::VectorXDecisionVariable& x_vars,
                      const solvers::VectorXDecisionVariable& t_vars,
                      const VectorXd& x, const VectorXd& t,
                      solvers::MathematicalProgram* prog,
                      const std::vector<Binding<Constraint>>& constraints) {
  const double tol = 0;
  prog->SetInitialGuess(x_vars, x);
  prog->SetInitialGuess(t_vars, t);
  return prog->CheckSatisfiedAtInitialGuess(constraints, tol);
}

GTEST_TEST(HPolyhedronTest, NonnegativeScalingTest2) {
  const Vector3d lb{1, 1, 1}, ub{2, 3, 4};
  HPolyhedron H = HPolyhedron::MakeBox(lb, ub);

  MathematicalProgram prog;
  MatrixXd A(3, 2);
  // clang-format off
  A << 1, 0,
       0, 1,
       2, 0;
  // clang-format on
  Vector3d b = Vector3d::Zero();
  auto x = prog.NewContinuousVariables(2, "x");
  Vector2d c(1, -1);
  double d = 0;
  auto t = prog.NewContinuousVariables(2, "t");

  std::vector<Binding<Constraint>> constraints =
      H.AddPointInNonnegativeScalingConstraints(&prog, A, b, c, d, x, t);

  EXPECT_EQ(constraints.size(), 2);

  EXPECT_TRUE(PointInScaledSet(x, t, 0.99 * ub.head(2), Vector2d(1.0, 0), &prog,
                               constraints));
  EXPECT_TRUE(PointInScaledSet(x, t, 0.99 * ub.head(2), Vector2d(0, -1.0),
                               &prog, constraints));
  EXPECT_FALSE(PointInScaledSet(x, t, 1.01 * ub.head(2), Vector2d(1.0, 0),
                                &prog, constraints));
  EXPECT_FALSE(PointInScaledSet(x, t, 1.01 * ub.head(2), Vector2d(0, -1.0),
                                &prog, constraints));
  EXPECT_FALSE(PointInScaledSet(x, t, 0.99 * ub.head(2), Vector2d(-0.01, 0),
                                &prog, constraints));
  EXPECT_FALSE(PointInScaledSet(x, t, 0.99 * ub.head(2), Vector2d(0, -0.01),
                                &prog, constraints));
  EXPECT_TRUE(PointInScaledSet(x, t, 0.49 * ub.head(2), Vector2d(0.5, 0), &prog,
                               constraints));
  EXPECT_TRUE(PointInScaledSet(x, t, 0.49 * ub.head(2), Vector2d(0, -0.5),
                               &prog, constraints));
  EXPECT_FALSE(PointInScaledSet(x, t, 0.51 * ub.head(2), Vector2d(0.5, 0),
                                &prog, constraints));
  EXPECT_FALSE(PointInScaledSet(x, t, 0.51 * ub.head(2), Vector2d(0, -0.5),
                                &prog, constraints));
  EXPECT_TRUE(PointInScaledSet(x, t, 1.99 * ub.head(2), Vector2d(2.0, 0), &prog,
                               constraints));
  EXPECT_TRUE(PointInScaledSet(x, t, 1.99 * ub.head(2), Vector2d(0, -2.0),
                               &prog, constraints));
  EXPECT_FALSE(PointInScaledSet(x, t, 2.01 * ub.head(2), Vector2d(2.0, 0),
                                &prog, constraints));
  EXPECT_FALSE(PointInScaledSet(x, t, 2.01 * ub.head(2), Vector2d(0, -2.0),
                                &prog, constraints));
}

GTEST_TEST(HPolyhedronTest, IsBounded) {
  Vector4d lb, ub;
  lb << -1, -3, -5, -2;
  ub << 2, 4, 5.4, 3;
  HPolyhedron H = HPolyhedron::MakeBox(lb, ub);
  EXPECT_TRUE(H.IsBounded());
}

GTEST_TEST(HPolyhedronTest, IsBounded2) {
  // Box with zero volume.
  const Vector2d lb{1, -3}, ub{1, 3};
  HPolyhedron H = HPolyhedron::MakeBox(lb, ub);
  EXPECT_TRUE(H.IsBounded());
}

GTEST_TEST(HPolyhedronTest, IsBounded3) {
  // Unbounded (2 inequalities in 3 dimensions).
  HPolyhedron H(MatrixXd::Identity(2, 3), Vector2d::Ones());
  EXPECT_FALSE(H.IsBounded());
}

GTEST_TEST(HPolyhedronTest, IsBounded4) {
  // Unbounded (A is low rank).
  Matrix3d A;
  // clang-format off
  A << 1, 2, 3,
       1, 2, 3,
       0, 0, 1;
  // clang-format on
  HPolyhedron H(A, Vector3d::Ones());
  EXPECT_FALSE(H.IsBounded());
}

GTEST_TEST(HPolyhedronTest, IsBoundedEmptyPolyhedron) {
  Eigen::MatrixXd A_infeasible{3, 3};
  // clang-format off
  A_infeasible << 1, -1, 0,
                  -1, 0, 1,
                  0, 1, -1;
  // clang-format on
  HPolyhedron H(A_infeasible, -Vector3d::Ones());
  EXPECT_TRUE(H.IsEmpty());
  EXPECT_FALSE(H.MaybeGetFeasiblePoint().has_value());
}

GTEST_TEST(HPolyhedronTest, CartesianPowerTest) {
  // First test the concept. If x ∈ H, then [x; x]  ∈ H x H and
  // [x; x; x]  ∈ H x H x H.
  MatrixXd A{4, 2};
  A << MatrixXd::Identity(2, 2), -MatrixXd::Identity(2, 2);
  VectorXd b = VectorXd::Ones(4);
  HPolyhedron H(A, b);
  VectorXd x = VectorXd::Zero(2);
  EXPECT_TRUE(H.PointInSet(x));
  EXPECT_TRUE(H.CartesianPower(2).PointInSet((VectorXd(4) << x, x).finished()));
  EXPECT_TRUE(
      H.CartesianPower(3).PointInSet((VectorXd(6) << x, x, x).finished()));

  // Now test the HPolyhedron-specific behavior.
  MatrixXd A_1{2, 3};
  MatrixXd A_2{4, 6};
  MatrixXd A_3{6, 9};
  VectorXd b_1{2};
  VectorXd b_2{4};
  VectorXd b_3{6};
  MatrixXd zero = MatrixXd::Zero(2, 3);
  // clang-format off
  A_1 << 1, 2, 3,
         4, 5, 6;
  b_1 << 1, 2;
  A_2 <<  A_1, zero,
         zero,  A_1;
  b_2 << b_1, b_1;
  A_3 <<  A_1, zero, zero,
         zero,  A_1, zero,
         zero, zero,  A_1;
  b_3 << b_1, b_1, b_1;
  // clang-format on
  HPolyhedron H_1(A_1, b_1);
  HPolyhedron H_2 = H_1.CartesianPower(2);
  HPolyhedron H_3 = H_1.CartesianPower(3);
  EXPECT_TRUE(CompareMatrices(H_2.A(), A_2));
  EXPECT_TRUE(CompareMatrices(H_2.b(), b_2));
  EXPECT_TRUE(CompareMatrices(H_3.A(), A_3));
  EXPECT_TRUE(CompareMatrices(H_3.b(), b_3));
}

GTEST_TEST(HPolyhedronTest, CartesianProductTest) {
  HPolyhedron H_A = HPolyhedron::MakeUnitBox(2);
  VectorXd x_A = VectorXd::Zero(2);
  EXPECT_TRUE(H_A.PointInSet(x_A));

  HPolyhedron H_B = HPolyhedron::MakeBox(Vector2d(2, 2), Vector2d(4, 4));
  VectorXd x_B = 3 * VectorXd::Ones(2);
  EXPECT_TRUE(H_B.PointInSet(x_B));

  HPolyhedron H_C = H_A.CartesianProduct(H_B);
  VectorXd x_C{x_A.size() + x_B.size()};
  x_C << x_A, x_B;
  EXPECT_TRUE(H_C.PointInSet(x_C));

  ASSERT_TRUE(H_C.MaybeGetFeasiblePoint().has_value());
  EXPECT_TRUE(H_C.PointInSet(H_C.MaybeGetFeasiblePoint().value()));
}

GTEST_TEST(HPolyhedronTest, AxisAlignedContainment) {
  Vector2d lower_limit = -Vector2d::Ones();
  Vector2d upper_limit = Vector2d::Ones();
  double scale{0.25};

  HPolyhedron outer = HPolyhedron::MakeBox(lower_limit, upper_limit);
  HPolyhedron inner =
      HPolyhedron::MakeBox(scale * lower_limit, scale * upper_limit);

  EXPECT_TRUE(inner.ContainedIn(outer));
  EXPECT_FALSE(outer.ContainedIn(inner));
}

GTEST_TEST(HPolyhedronTest, L1BallContainsInfinityBall3D) {
  HPolyhedron L1_ball = HPolyhedron::MakeL1Ball(3);
  HPolyhedron Linfty_ball = HPolyhedron::MakeUnitBox(3);

  EXPECT_TRUE(L1_ball.ContainedIn(Linfty_ball));
  EXPECT_FALSE(Linfty_ball.ContainedIn(L1_ball));
}

GTEST_TEST(HPolyhedronTest, L1BallIrredundantIntersectionInfinityBall3D) {
  HPolyhedron L1_ball = HPolyhedron::MakeL1Ball(3);
  HPolyhedron Linfty_ball = HPolyhedron::MakeUnitBox(3);

  HPolyhedron IntersectionBall = L1_ball.Intersection(Linfty_ball, true);
  EXPECT_TRUE(CompareMatrices(L1_ball.A(), IntersectionBall.A()));
  EXPECT_TRUE(CompareMatrices(L1_ball.b(), IntersectionBall.b()));
}

GTEST_TEST(HPolyhedronTest, OffsetIrredundantBoxes) {
  Vector2d left_box_lower = {-1, -1};
  Vector2d left_box_upper = {0.25, 1};
  HPolyhedron left_box = HPolyhedron::MakeBox(left_box_lower, left_box_upper);

  Vector2d right_box_lower = {-0.25, -1};
  Vector2d right_box_upper = {1, 1};
  HPolyhedron right_box =
      HPolyhedron::MakeBox(right_box_lower, right_box_upper);

  HPolyhedron intersection_right_into_left =
      left_box.Intersection(right_box, true);
  HPolyhedron intersection_left_into_right =
      right_box.Intersection(left_box, true);

  MatrixXd A_right_into_left_expected(5, 2);
  VectorXd b_right_into_left_expected(5);
  MatrixXd A_left_into_right_expected(5, 2);
  VectorXd b_left_into_right_expected(5);

  A_right_into_left_expected.topRows(4) = left_box.A();
  b_right_into_left_expected.topRows(4) = left_box.b();
  A_left_into_right_expected.topRows(4) = right_box.A();
  b_left_into_right_expected.topRows(4) = right_box.b();

  A_right_into_left_expected.row(4) = right_box.A().row(2);
  b_right_into_left_expected.row(4) = right_box.b().row(2);

  A_left_into_right_expected.row(4) = left_box.A().row(0);
  b_left_into_right_expected.row(4) = left_box.b().row(0);

  EXPECT_TRUE(CompareMatrices(A_right_into_left_expected,
                              intersection_right_into_left.A()));
  EXPECT_TRUE(CompareMatrices(b_right_into_left_expected,
                              intersection_right_into_left.b()));

  EXPECT_TRUE(CompareMatrices(A_left_into_right_expected,
                              intersection_left_into_right.A()));
  EXPECT_TRUE(CompareMatrices(b_left_into_right_expected,
                              intersection_left_into_right.b()));
}

GTEST_TEST(HPolyhedronTest, ContainedIn) {
  // Checks Contained in with tolerance.
  const HPolyhedron small_polyhedron(Eigen::RowVector2d(1, 1), Vector1d(2));
  const HPolyhedron large_polyhedron(Eigen::RowVector2d(1, 1), Vector1d(3));
  EXPECT_FALSE(large_polyhedron.ContainedIn(small_polyhedron, 0));
  // We think the containment is true if we relax the tolerance.
  EXPECT_TRUE(large_polyhedron.ContainedIn(small_polyhedron, 1.1));
}

GTEST_TEST(HPolyhedronTest, IrredundantBallIntersectionContainsBothOriginal) {
  HPolyhedron L1_ball = HPolyhedron::MakeL1Ball(3);
  HPolyhedron Linfty_ball = HPolyhedron::MakeUnitBox(3);

  HPolyhedron IrredL1intoLinf = Linfty_ball.Intersection(L1_ball, true);
  HPolyhedron IrredLinfintoL1 = L1_ball.Intersection(Linfty_ball, true);

  EXPECT_TRUE(IrredL1intoLinf.ContainedIn(L1_ball, 3E-7));
  EXPECT_TRUE(IrredL1intoLinf.ContainedIn(Linfty_ball));
  EXPECT_TRUE(IrredLinfintoL1.ContainedIn(L1_ball));
  EXPECT_TRUE(IrredLinfintoL1.ContainedIn(Linfty_ball));
}

GTEST_TEST(HPolyhedronTest, ReduceL1LInfBallIntersection) {
  HPolyhedron L1_ball = HPolyhedron::MakeL1Ball(3);
  HPolyhedron Linfty_ball = HPolyhedron::MakeUnitBox(3);

  MatrixXd A_int(L1_ball.A().rows() + Linfty_ball.A().rows(), 3);
  MatrixXd b_int(A_int.rows(), 1);
  A_int.topRows(L1_ball.A().rows()) = L1_ball.A();
  b_int.topRows(L1_ball.b().rows()) = L1_ball.b();
  A_int.bottomRows(Linfty_ball.A().rows()) = Linfty_ball.A();
  b_int.bottomRows(Linfty_ball.b().rows()) = Linfty_ball.b();
  HPolyhedron polyhedron_to_reduce(A_int, b_int);
  const double tol = 1E-7;
  const auto redundant_indices = polyhedron_to_reduce.FindRedundant(tol);
  // Removed Linfty_ball.
  std::set<int> redundant_indices_expected;
  for (int i = 0; i < Linfty_ball.A().rows(); ++i) {
    redundant_indices_expected.emplace(i + L1_ball.A().rows());
  }
  EXPECT_EQ(redundant_indices, redundant_indices_expected);
  HPolyhedron reduced_polyhedron = polyhedron_to_reduce.ReduceInequalities(tol);

  EXPECT_TRUE(CompareMatrices(reduced_polyhedron.A(), L1_ball.A()));
  EXPECT_TRUE(CompareMatrices(reduced_polyhedron.b(), L1_ball.b()));
}

GTEST_TEST(HPolyhedronTest, ReduceToInfeasibleSet) {
  Eigen::MatrixXd A{5, 3};
  Eigen::VectorXd b{5};
  // Rows 1-3 define an infeasible set of inequalities.
  // clang-format off
  A << 1, 0, 0,
       1, -1, 0,
       -1, 0, 1,
       0, 1, -1,
       0, 0, -1;
  b << 1, -1, -1, -1, 0;
  // clang-format on

  HPolyhedron H{A, b};
  HPolyhedron H_reduced = H.ReduceInequalities();

  EXPECT_TRUE(H.IsEmpty());
  EXPECT_TRUE(H_reduced.IsEmpty());
  EXPECT_FALSE(H.MaybeGetFeasiblePoint().has_value());
  EXPECT_FALSE(H_reduced.MaybeGetFeasiblePoint().has_value());
}

GTEST_TEST(HPolyhedronTest, IsEmptyMinimalInequalitySet) {
  Eigen::MatrixXd A_infeasible{3, 3};
  Eigen::VectorXd b_infeasible{3};
  // clang-format off
  A_infeasible << 1, -1, 0,
                  -1, 0, 1,
                  0, 1, -1;
  b_infeasible << -1, -1, -1;
  // clang-format on

  HPolyhedron H{A_infeasible, b_infeasible};
  EXPECT_TRUE(H.IsEmpty());
}

GTEST_TEST(HPolyhedronTest, IsEmptyNonMinimalInequalitySet) {
  Eigen::MatrixXd A{5, 3};
  Eigen::VectorXd b{5};
  // clang-format off
  A << 1, 0, 0,
       0, 0, -1,
       1, -1, 0,
       -1, 0, 1,
       0, 1, -1;
  b << 1, 0, -1, -1, -1;
  // clang-format on

  HPolyhedron H{A, b};
  EXPECT_TRUE(H.IsEmpty());
}

GTEST_TEST(HPolyhedronTest, IsEmptyUnboundedHPolyhedron) {
  Eigen::MatrixXd A{2, 2};
  Eigen::VectorXd b{2};
  A << 1, 0, -1, 0;  // only restrict the first coordinate
  b << 1, 1;
  HPolyhedron H{A, b};
  EXPECT_FALSE(H.IsEmpty());
}

GTEST_TEST(HPolyhedronTest, IsEmptyBoundedHPolyhedron) {
  HPolyhedron H = HPolyhedron::MakeUnitBox(2);
  EXPECT_FALSE(H.IsEmpty());
}

GTEST_TEST(HPolyhedronTest, IntersectionTest) {
  HPolyhedron H_A = HPolyhedron::MakeUnitBox(2);
  HPolyhedron H_B = HPolyhedron::MakeBox(Vector2d(0, 0), Vector2d(2, 2));
  HPolyhedron H_C = H_A.Intersection(H_B);

  Vector2d x_C(0.5, 0.5);
  EXPECT_TRUE(H_A.PointInSet(x_C));
  EXPECT_TRUE(H_B.PointInSet(x_C));
  EXPECT_TRUE(H_C.PointInSet(x_C));

  Vector2d x_A(-0.5, -0.5);
  EXPECT_TRUE(H_A.PointInSet(x_A));
  EXPECT_FALSE(H_B.PointInSet(x_A));
  EXPECT_FALSE(H_C.PointInSet(x_A));

  Vector2d x_B(1.5, 1.5);
  EXPECT_FALSE(H_A.PointInSet(x_B));
  EXPECT_TRUE(H_B.PointInSet(x_B));
  EXPECT_FALSE(H_C.PointInSet(x_B));

  ASSERT_TRUE(H_C.MaybeGetFeasiblePoint().has_value());
  EXPECT_TRUE(H_C.PointInSet(H_C.MaybeGetFeasiblePoint().value()));
}

GTEST_TEST(HPolyhedronTest, PontryaginDifferenceTestAxisAligned) {
  const HPolyhedron H_A = HPolyhedron::MakeUnitBox(2);
  const HPolyhedron H_B = HPolyhedron::MakeBox(Vector2d(0, 0), Vector2d(1, 1));
  const HPolyhedron H_C = H_A.PontryaginDifference(H_B);
  const HPolyhedron H_C_expected =
      HPolyhedron::MakeBox(Vector2d{-1, -1}, Vector2d{0, 0});

  EXPECT_TRUE(CompareMatrices(H_C.A(), H_C_expected.A(), 1e-8));
  EXPECT_TRUE(CompareMatrices(H_C.b(), H_C_expected.b(), 1e-8));

  ASSERT_TRUE(H_C.MaybeGetFeasiblePoint().has_value());
  EXPECT_TRUE(H_C.PointInSet(H_C.MaybeGetFeasiblePoint().value()));
}

GTEST_TEST(HPolyhedronTest, PontryaginDifferenceTestSquareTriangle) {
  HPolyhedron H_A = HPolyhedron::MakeUnitBox(2);

  Matrix<double, 3, 2> A_B;
  Vector<double, 3> b_B;
  // clang-format off
  A_B << -1, 0,
          0, -1,
          1, 1;
  b_B << 0, 0, 1;
  // clang-format on
  // right triangle with vertices [0,0], [1,0], [0,1]
  const HPolyhedron H_B{A_B, b_B};

  const HPolyhedron H_C = H_A.PontryaginDifference(H_B);

  const HPolyhedron H_C_expected =
      HPolyhedron::MakeBox(Vector2d{-1, -1}, Vector2d{0, 0});

  EXPECT_TRUE(CompareMatrices(H_C.A(), H_C_expected.A(), 1e-8));
  EXPECT_TRUE(CompareMatrices(H_C.b(), H_C_expected.b(), 1e-8));
}

GTEST_TEST(HPolyhedronTest, PontryaginDifferenceTestNonAxisAligned) {
  // L1 box scaled to have corners at 0.5 instead of 1; it is intentionally not
  // axis aligned in this test
  HPolyhedron L1_ball = HPolyhedron::MakeL1Ball(3);
  const HPolyhedron H_A = HPolyhedron::MakeUnitBox(3);

  const HPolyhedron H_B{L1_ball.A(), 0.5 * L1_ball.b()};

  const HPolyhedron H_C = H_A.PontryaginDifference(H_B);

  const HPolyhedron H_C_expected =
      HPolyhedron::MakeBox(Vector3d::Constant(-0.5), Vector3d::Constant(0.5));

  EXPECT_TRUE(CompareMatrices(H_C.A(), H_C_expected.A(), 1e-8));
  EXPECT_TRUE(CompareMatrices(H_C.b(), H_C_expected.b(), 1e-8));
}

GTEST_TEST(HPolyhedronTest, UniformSampleTest1) {
  Matrix<double, 4, 2> A;
  Vector4d b;
  // clang-format off
  A << -2, -1,  // 2x + y ≥ 4
        2,  1,  // 2x + y ≤ 6
       -1,  2,  // x - 2y ≥ 2
        1, -2;  // x - 2y ≤ 8
  b << -4, 6, -2, 8;
  // clang-format on
  HPolyhedron H(A, b);

  // Draw random samples.
  RandomGenerator generator(1234);
  const int N{10000};
  MatrixXd samples(2, N);
  const int mixing_steps{7};
  samples.col(0) = H.UniformSample(&generator, mixing_steps);
  for (int i = 1; i < N; ++i) {
    samples.col(i) =
        H.UniformSample(&generator, samples.col(i - 1), mixing_steps);
  }

  // Provide a visualization of the points.
  {
    std::shared_ptr<Meshcat> meshcat = geometry::GetTestEnvironmentMeshcat();
    meshcat->SetProperty("/Background", "visible", false);
    perception::PointCloud cloud(N);
    cloud.mutable_xyzs().topRows<2>() = samples.cast<float>();
    cloud.mutable_xyzs().bottomRows<1>().setZero();
    meshcat->SetObject("samples", cloud, 0.01, Rgba(0, 0, 1));

    common::MaybePauseForUser();
  }

  // Check that they are all in the polyhedron.
  for (int i = 0; i < A.rows(); ++i) {
    EXPECT_LE((A.row(i) * samples).maxCoeff(), b(i));
  }

  const double kTol = 0.05 * N;
  // Check that approximately half of them satisfy 2x+y ≥ 5.
  EXPECT_NEAR(((2 * samples.row(0) + samples.row(1)).array() >= 5.0).count(),
              0.5 * N, kTol);

  // Check that approximately half of them satisfy x - 2y ≥ 5.
  EXPECT_NEAR(((samples.row(0) - 2 * samples.row(1)).array() >= 5.0).count(),
              0.5 * N, kTol);

  // Check that an off-center box gets the number of samples proportional to
  // its (relative) volume. H is a rotated box with volume 1 x 2.5 = 2.5. We'll
  // check the box: 3 ≤ x ≤ 3.5, -1.5 ≤ y ≤ -1, which has volume .5 x .5 = .25.
  EXPECT_NEAR((samples.row(0).array() >= 3 && samples.row(0).array() <= 3.5 &&
               samples.row(1).array() >= -1.5 && samples.row(1).array() <= -1)
                  .count(),
              N / 10, kTol);
}

// Test the case where the sample point is outside the region, but the max
// threshold can be smaller than the min threshold. (This was a bug uncovered
// by hammering on this code from IRIS).
GTEST_TEST(HPolyhedronTest, UniformSampleTest2) {
  Matrix<double, 5, 2> A;
  Matrix<double, 5, 1> b;
  // clang-format off
  A <<  1,  0,  // x ≤ 1
        0,  1,  // y ≤ 1
       -1,  0,  // x ≥ -1
        0, -1,  // y ≥ -1
       -1,  0,  // x ≥ 0
  b << 1, 1, 1, 1, 0;
  // clang-format on
  HPolyhedron H(A, b);

  // Draw random samples.
  RandomGenerator generator(1234);
  // Use a seed that is outside the set (because x ≤ 0), but still inside the
  // [-1, 1] unit box (so the line search in all directions returns finite
  // values). It throws when the hit and run direction intersects x=0 outside
  // of the unit box.
  const Vector2d seed{-0.5, 0.9};
  // Make sure that random samples either return a point in the set (because
  // they were lucky) or throw.  Previously, the method could return a point
  // outside the set.
  int num_throws = 0;
  int num_success = 0;
  for (int i = 0; i < 10; ++i) {
    try {
      const Vector2d sample = H.UniformSample(&generator, seed);
      EXPECT_TRUE(H.PointInSet(sample, 1e-12));
      ++num_success;
    } catch (const std::exception& err) {
      ++num_throws;
      EXPECT_NE(
          std::string(err.what())
              .find("Hit and Run algorithm failed to find a feasible point"),
          std::string::npos);
    }
  }
  // Make sure both paths were touched.
  EXPECT_GT(num_throws, 0);
  EXPECT_GT(num_success, 0);
}

// Test that the argument mixing_steps is working by sampling three points: A
// with 5 mixing steps starting from the Chebyshev center, B starting from the
// same random seed as A, but with 2 mixing steps, and C starting from B with 2
// mixing steps. We expect A==C but A!=B.
GTEST_TEST(HPolyhedronTest, UniformSampleTest3) {
  Matrix<double, 4, 2> D;
  Vector4d e;
  // clang-format off
  D << -2, -1,  // 2x + y ≥ 4
        2,  1,  // 2x + y ≤ 6
       -1,  2,  // x - 2y ≥ 2
        1, -2;  // x - 2y ≤ 8
  e << -4, 6, -2, 8;
  // clang-format on
  HPolyhedron H(D, e);

  // Draw random samples.
  RandomGenerator generator(1234);
  Vector2d A = H.UniformSample(&generator, 5);
  RandomGenerator generator2(1234);
  Vector2d B = H.UniformSample(&generator2, 2);
  Vector2d C = H.UniformSample(&generator2, B, 3);
  const double kTol = 1e-7;

  EXPECT_TRUE(CompareMatrices(A, C, kTol));
  EXPECT_FALSE(CompareMatrices(A, B, kTol));
}

// Test that we can draw samples from not-full-dimensional HPolyhedra
GTEST_TEST(HPolyhedronTest, UniformSampleTest4) {
  Matrix<double, 2, 2> points;
  // clang-format off
  points << 0, 1,
            0, 1;
  // clang-format on
  VPolytope V(points);
  HPolyhedron H(V);
  const double kTol = 1e-7;

  // Verify that without passing in the basis of the affine hull,
  // trying to draw a uniform sample just gives us previous_sample,
  // since it is not full-dimensional.
  Vector2d point(0.5, 0.5);

  RandomGenerator generator(1234);
  VectorXd point_A = H.UniformSample(&generator, point, 10);
  EXPECT_TRUE(CompareMatrices(point, point_A, kTol));

  // Compute the affine hull, and use this to draw samples.
  AffineSubspace as(H);
  MatrixXd basis = as.basis();
  VectorXd point_B = H.UniformSample(&generator, point, 10, basis);
  EXPECT_FALSE(CompareMatrices(point, point_B, kTol));

  // Check that a subspace of incompatible shape throws. In this case,
  // the subspace has four rows, which does not equal the ambient
  // dimension of the HPolyhedron, which is two.
  Matrix<double, 4, 1> bad_basis;
  bad_basis << 0, 1, 2, 3;
  EXPECT_THROW(H.UniformSample(&generator, point, 1, bad_basis),
               std::exception);
}

GTEST_TEST(HPolyhedronTest, Serialize) {
  const HPolyhedron H = HPolyhedron::MakeL1Ball(3);
  const std::string yaml = yaml::SaveYamlString(H);
  const auto H2 = yaml::LoadYamlString<HPolyhedron>(yaml);
  EXPECT_EQ(H.ambient_dimension(), H2.ambient_dimension());
  EXPECT_TRUE(CompareMatrices(H.A(), H2.A()));
  EXPECT_TRUE(CompareMatrices(H.b(), H2.b()));
}

GTEST_TEST(HPolyhedronTest, SimplifyByIncrementalFaceTranslation1) {
  // Test a case where the expected simplified polytope is known:
  // The circumbody is a square with the top-right and bottom-left corners cut
  // off (6 faces).  The inbody should remove the two diagonal faces by scaling
  // the left and right faces in by (min_volume_ratio) ^ (1.0 /
  // ambient_dimension()).
  const double kConstraintTol = 1e-6;
  Eigen::Matrix<double, 6, 2> A;
  // clang-format off
  A << 1, 0,  // x <= 2
       -1, 0,  // -x <= 2
       0, 1,  // y <= 2
       0, -1,  // -y <= 2
       1, 1,  // x + y <= 3.5
       -1, -1;  // -x - y <= 3.5
  // clang-format on
  Eigen::VectorXd b(6);
  b << 2, 2, 2, 2, 3.5, 3.5;
  const HPolyhedron circumbody = HPolyhedron(A, b);
  const double min_volume_ratio = 0.1;

  const HPolyhedron inbody =
      circumbody.SimplifyByIncrementalFaceTranslation(min_volume_ratio, false);

  Eigen::Matrix<double, 4, 2> A_expected;
  // clang-format off
  A_expected << 1, 0,  // x <= 2 * min_volume_ratio ^ (1 / dimension)
                -1, 0,  // -x <= 2 * min_volume_ratio ^ (1 / dimension)
                0, 1,  // y <= 2
                0, -1;  // -y <= 2
  // clang-format on
  Eigen::VectorXd b_expected(4);
  b_expected << 2 * std::pow(min_volume_ratio, 0.5),
      2 * std::pow(min_volume_ratio, 0.5), 2, 2;
  const HPolyhedron inbody_expected = HPolyhedron(A_expected, b_expected);

  EXPECT_TRUE(inbody_expected.ContainedIn(inbody, kConstraintTol));
  EXPECT_TRUE(inbody.ContainedIn(inbody_expected, kConstraintTol));
}

GTEST_TEST(HPolyhedronTest, SimplifyByIncrementalFaceTranslation2) {
  // Test that if `min_volume_ratio` = 1, and the circumbody has no redundant
  // faces, the circumbody is returned unchanged.
  const int kNumFaces = 20;
  const double kConstraintTol = 1e-6;

  // Create a polygon in 2D with `kNumFaces` faces from unit circle tangents.
  MatrixXd A(kNumFaces, 2);
  for (int row = 0; row < kNumFaces; ++row) {
    A.row(row) << std::cos(2 * M_PI * row / kNumFaces),
        std::sin(2 * M_PI * row / kNumFaces);
  }
  const VectorXd b = VectorXd::Ones(kNumFaces);
  const HPolyhedron circumbody = HPolyhedron(A, b);
  const double min_volume_ratio = 1.0;

  const HPolyhedron inbody =
      circumbody.SimplifyByIncrementalFaceTranslation(min_volume_ratio, false);

  EXPECT_TRUE(inbody.ContainedIn(circumbody, kConstraintTol));
  EXPECT_TRUE(circumbody.ContainedIn(inbody, kConstraintTol));
}

GTEST_TEST(HPolyhedronTest, SimplifyByIncrementalFaceTranslation3) {
  // Test simplification of `circumbody` subject to keeping whole intersection
  // with `intersecting_polytope`, with 0 `intersection_padding`.
  const int kNumFaces = 20;
  const double kConstraintTol = 1e-6;

  // Create a polygon in 2D with `kNumFaces` faces from unit circle tangents.
  MatrixXd A(kNumFaces, 2);
  for (int row = 0; row < kNumFaces; ++row) {
    A.row(row) << std::cos(2 * M_PI * row / kNumFaces),
        std::sin(2 * M_PI * row / kNumFaces);
  }
  const VectorXd b = VectorXd::Ones(kNumFaces);
  const HPolyhedron circumbody = HPolyhedron(A, b);
  const VPolytope circumbody_V(circumbody);  // For volume calculations.
  const double min_volume_ratio = 0.1;

  // Create a triangle polytope that intersects with the circumbody.
  Eigen::Matrix<double, 2, 3> verts;
  // clang-format off
  verts << 0, 1, -1,
           0.8, 2, 2;
  // clang-format on
  const HPolyhedron intersecting_polytope = HPolyhedron(VPolytope(verts));
  const std::vector<HPolyhedron> intersecting_polytopes = {
      intersecting_polytope};
  const HPolyhedron inbody = circumbody.SimplifyByIncrementalFaceTranslation(
      min_volume_ratio, false, 10, Eigen::MatrixXd(), intersecting_polytopes,
      true, 0);
  EXPECT_TRUE(inbody.ContainedIn(circumbody, kConstraintTol));
  EXPECT_GE(VPolytope(inbody).CalcVolume() / circumbody_V.CalcVolume(),
            min_volume_ratio);
  EXPECT_LE(inbody.b().rows(), circumbody.b().rows());
  // Check if intersection is still contained.
  EXPECT_TRUE((circumbody.Intersection(intersecting_polytope)
                   .ContainedIn(inbody, kConstraintTol)));
}

GTEST_TEST(HPolyhedronTest, SimplifyByIncrementalFaceTranslation4) {
  // Test that non-zero `intersection_padding` does not break anything from the
  // last test.
  const int kNumFaces = 20;
  const double kConstraintTol = 1e-6;

  // Create a polygon in 2D with `kNumFaces` faces from unit circle tangents.
  MatrixXd A(kNumFaces, 2);
  for (int row = 0; row < kNumFaces; ++row) {
    A.row(row) << std::cos(2 * M_PI * row / kNumFaces),
        std::sin(2 * M_PI * row / kNumFaces);
  }
  const VectorXd b = VectorXd::Ones(kNumFaces);
  const HPolyhedron circumbody = HPolyhedron(A, b);
  const VPolytope circumbody_V(circumbody);  // For volume calculations.
  const double min_volume_ratio = 0.1;

  // Create a triangle polytope that intersects with the circumbody.
  Eigen::Matrix<double, 2, 3> verts;
  // clang-format off
  verts << 0, 1, -1,
           0.8, 2, 2;
  // clang-format on
  const HPolyhedron intersecting_polytope = HPolyhedron(VPolytope(verts));
  const std::vector<HPolyhedron> intersecting_polytopes = {
      intersecting_polytope};

  const HPolyhedron inbody = circumbody.SimplifyByIncrementalFaceTranslation(
      min_volume_ratio, false, 10, Eigen::MatrixXd(), intersecting_polytopes,
      true, 0.1);
  EXPECT_TRUE(inbody.ContainedIn(circumbody, kConstraintTol));
  EXPECT_GE(VPolytope(inbody).CalcVolume() / circumbody_V.CalcVolume(),
            min_volume_ratio);
  EXPECT_LE(inbody.b().rows(), circumbody.b().rows());
  // Check if intersection is still contained.
  EXPECT_TRUE((circumbody.Intersection(intersecting_polytope)
                   .ContainedIn(inbody, kConstraintTol)));
}

GTEST_TEST(HPolyhedronTest, SimplifyByIncrementalFaceTranslation5) {
  // Test with affine transformation.
  const int kNumFaces = 20;
  // Containment constraint needs higher tolerance after affine transformation.
  const double kAffineTransformationConstraintTol = 1e-4;

  // Create a polygon in 2D with `kNumFaces` faces from unit circle tangents.
  MatrixXd A(kNumFaces, 2);
  for (int row = 0; row < kNumFaces; ++row) {
    A.row(row) << std::cos(2 * M_PI * row / kNumFaces),
        std::sin(2 * M_PI * row / kNumFaces);
  }
  const VectorXd b = VectorXd::Ones(kNumFaces);
  const HPolyhedron circumbody = HPolyhedron(A, b);
  const VPolytope circumbody_V(circumbody);  // For volume calculations.
  const double min_volume_ratio = 0.1;

  // Create a triangle polytope that intersects with the circumbody.
  Eigen::Matrix<double, 2, 3> verts;
  // clang-format off
  verts << 0, 1, -1,
           0.8, 2, 2;
  // clang-format on
  const HPolyhedron intersecting_polytope = HPolyhedron(VPolytope(verts));
  const std::vector<HPolyhedron> intersecting_polytopes = {
      intersecting_polytope};
  const HPolyhedron inbody = circumbody.SimplifyByIncrementalFaceTranslation(
      min_volume_ratio, true, 10, Eigen::MatrixXd(), intersecting_polytopes,
      false);
  EXPECT_TRUE(
      inbody.ContainedIn(circumbody, kAffineTransformationConstraintTol));
  EXPECT_GE(VPolytope(inbody).CalcVolume() / circumbody_V.CalcVolume(),
            min_volume_ratio);
  EXPECT_LE(inbody.b().rows(), circumbody.b().rows());
  // We only expect to maintain part of the intersection, not to contain the
  // whole original intersection.
  EXPECT_TRUE(inbody.IntersectsWith(intersecting_polytope));
}

GTEST_TEST(HPolyhedronTest, SimplifyByIncrementalFaceTranslation6) {
  // Test with points to contain.
  const int kNumFaces = 20;
  const double kConstraintTol = 1e-6;

  // Create a polygon in 2D with `kNumFaces` faces from unit circle tangents.
  MatrixXd A(kNumFaces, 2);
  for (int row = 0; row < kNumFaces; ++row) {
    A.row(row) << std::cos(2 * M_PI * row / kNumFaces),
        std::sin(2 * M_PI * row / kNumFaces);
  }
  const VectorXd b = VectorXd::Ones(kNumFaces);
  const HPolyhedron circumbody = HPolyhedron(A, b);
  const VPolytope circumbody_V(circumbody);  // For volume calculations.
  const double min_volume_ratio = 0.1;

  Eigen::Matrix<double, 2, 3> points;
  // clang-format off
  points << 0, 0.7, -0.7,
           -1, 0.7, 0.7;
  // clang-format on
  const HPolyhedron inbody = circumbody.SimplifyByIncrementalFaceTranslation(
      min_volume_ratio, false, 10, points);
  EXPECT_TRUE(inbody.ContainedIn(circumbody, kConstraintTol));
  EXPECT_GE(VPolytope(inbody).CalcVolume() / circumbody_V.CalcVolume(),
            min_volume_ratio);
  EXPECT_LE(inbody.b().rows(), circumbody.b().rows());
  for (int i_point = 0; i_point < points.cols(); ++i_point) {
    EXPECT_TRUE(inbody.PointInSet(points.col(i_point), kConstraintTol));
  }
}

GTEST_TEST(HPolyhedronTest, SimplifyByIncrementalFaceTranslation7) {
  // Test with intersection, points to contain, and affine transformation
  const int kNumFaces = 20;
  const double kConstraintTol = 1e-6;
  // Containment constraint needs higher tolerance after affine transformation.
  const double kAffineTransformationConstraintTol = 1e-4;

  // Create a polygon in 2D with `kNumFaces` faces from unit circle tangents.
  MatrixXd A(kNumFaces, 2);
  for (int row = 0; row < kNumFaces; ++row) {
    A.row(row) << std::cos(2 * M_PI * row / kNumFaces),
        std::sin(2 * M_PI * row / kNumFaces);
  }
  const VectorXd b = VectorXd::Ones(kNumFaces);
  const HPolyhedron circumbody = HPolyhedron(A, b);
  const VPolytope circumbody_V(circumbody);  // For volume calculations.
  const double min_volume_ratio = 0.1;

  // Create a triangle polytope that intersects with the circumbody.
  Eigen::Matrix<double, 2, 3> verts;
  // clang-format off
  verts << 0, 1, -1,
           0.8, 2, 2;
  // clang-format on
  const HPolyhedron intersecting_polytope = HPolyhedron(VPolytope(verts));
  const std::vector<HPolyhedron> intersecting_polytopes = {
      intersecting_polytope};

  Eigen::Matrix<double, 2, 3> points;
  // clang-format off
  points << 0, 0.7, -0.7,
           -1, 0.7, 0.7;
  // clang-format on

  const HPolyhedron inbody = circumbody.SimplifyByIncrementalFaceTranslation(
      min_volume_ratio, true, 10, points, intersecting_polytopes, false);
  EXPECT_TRUE(
      inbody.ContainedIn(circumbody, kAffineTransformationConstraintTol));
  EXPECT_GE(VPolytope(inbody).CalcVolume() / circumbody_V.CalcVolume(),
            min_volume_ratio);
  EXPECT_LE(inbody.b().rows(), circumbody.b().rows());
  EXPECT_TRUE(inbody.IntersectsWith(intersecting_polytope));
  for (int i_point = 0; i_point < points.cols(); ++i_point) {
    EXPECT_TRUE(inbody.PointInSet(points.col(i_point), kConstraintTol));
  }
}

GTEST_TEST(HPolyhedronTest, MaximumVolumeInscribedAffineTransformationTest1) {
  const double kConstraintTol = 1e-4;
  const int kNumFaces = 4;
  const int kDimension = 2;

  // Test optimizing affine transformation of initially small polytope that is
  // not contained in circumbody.
  MatrixXd circumbody_A(kNumFaces, kDimension);
  for (int row = 0; row < kNumFaces; ++row) {
    circumbody_A.row(row) << std::cos(2 * M_PI * row / kNumFaces),
        std::sin(2 * M_PI * row / kNumFaces);
  }
  const VectorXd circumbody_b = VectorXd::Ones(kNumFaces);
  HPolyhedron circumbody = HPolyhedron(circumbody_A, circumbody_b);

  // clang-format off
  Eigen::Matrix<double, 2, 4> initial_polytope_verts;
  initial_polytope_verts << 1.4, 0.8, 0, 0.2,
                            0.5, 0.6, -0.2, 0.1;
  // clang-format on
  HPolyhedron initial_polytope = HPolyhedron(VPolytope(initial_polytope_verts));

  HPolyhedron inbody =
      initial_polytope.MaximumVolumeInscribedAffineTransformation(circumbody);

  // Check containment, and that volume increased.
  EXPECT_TRUE(inbody.ContainedIn(circumbody, kConstraintTol));
}

GTEST_TEST(HPolyhedronTest, MaximumVolumeInscribedAffineTransformationTest2) {
  const double kConstraintTol = 1e-4;
  const int kNumFaces = 4;
  const int kDimension = 2;

  // Test optimizing affine transformation of initially small polytope that is
  // contained in circumbody.
  MatrixXd circumbody_A(kNumFaces, kDimension);
  for (int row = 0; row < kNumFaces; ++row) {
    circumbody_A.row(row) << std::cos(2 * M_PI * row / kNumFaces),
        std::sin(2 * M_PI * row / kNumFaces);
  }
  const VectorXd circumbody_b = VectorXd::Ones(kNumFaces);
  HPolyhedron circumbody = HPolyhedron(circumbody_A, circumbody_b);

  // clang-format off
  Eigen::Matrix<double, 2, 4> initial_polytope_verts;
  initial_polytope_verts << 0.4, 0.8, 0, -0.5,
                            -0.2, 0.6, -0.2, 0.5;
  // clang-format on
  HPolyhedron initial_polytope = HPolyhedron(VPolytope(initial_polytope_verts));

  HPolyhedron inbody =
      initial_polytope.MaximumVolumeInscribedAffineTransformation(circumbody);

  // Check containment, and that volume increased.
  EXPECT_TRUE(inbody.ContainedIn(circumbody, kConstraintTol));
  EXPECT_GE(VPolytope(inbody).CalcVolume(),
            VPolytope(initial_polytope).CalcVolume());
}
}  // namespace optimization
}  // namespace geometry
}  // namespace drake
