/** **************************************************************************************************************************
* TomLeverTriangleTest encapsulates 28 JUnit tests for the Triangle class. Calling the triangeType method of a Triangle object
* provides the type of the triangle (i.e., equilateral, isosceles, or scale), or an error message.
*
* Student name:  Tom Lever
* Completion date: 06/07/21
*************************************************************************************************************************** */

import static org.junit.Assert.*;
import org.junit.Test;

public class TomLeverTriangleTest {


	/** --------------------------------------------------------------
	 * test1 tests that an equilateral triangle is identified as such.
	 -------------------------------------------------------------- */

	@Test
	public void test1(){
		// Triangle triangle = new Triangle("n", "n", "n");
		Triangle triangle = new Triangle("12","12","12");
		assertEquals("This is an equilateral triangle. ",triangle.triangleType());	
	}//end test
	

	/** ------------------------------------------------------------
	 * test2 tests that an isosceles triangle is identified as such.
	 ------------------------------------------------------------ */

	@Test
	public void test2(){
		Triangle triangle = new Triangle("3","3","5");
		assertEquals("This is an isosceles triangle. ",triangle.triangleType());
	}//end test
	

	/** ---------------------------------------------------------
	 * test3 tests that a scalene triangle is identified as such.
	 --------------------------------------------------------- */

	@Test
	public void test3(){
		Triangle triangle = new Triangle("4","5","6");
		assertEquals("This is a scalene triangle. ", triangle.triangleType());
		// expected value: "This is a scalene triangle. "  
		// value returned from the method: triangle.triangleType()
	}//end test
	

	/** -------------------------------------------------------------------------------------------------
	 * test4 through test10 are seven tests of order of side lengths for isosceles and scalene triangles.
	 * It turns out that order of side lengths does not matter for three "toy" triangles.
	 ------------------------------------------------------------------------------------------------- */

	@Test
	public void test4(){
		Triangle triangle = new Triangle("3","5","3");
		assertEquals("This is an isosceles triangle. ",triangle.triangleType());
	}//end test

	@Test
	public void test5(){
		Triangle triangle = new Triangle("5","3","3");
		assertEquals("This is an isosceles triangle. ",triangle.triangleType());
	}//end test
	
	@Test
	public void test6(){
		Triangle triangle = new Triangle("4","6","5");
		assertEquals("This is a scalene triangle. ", triangle.triangleType());
	}//end test
	
	@Test
	public void test7(){
		Triangle triangle = new Triangle("5","4","6");
		assertEquals("This is a scalene triangle. ", triangle.triangleType());
	}//end test
	
	@Test
	public void test8(){
		Triangle triangle = new Triangle("5","6","4");
		assertEquals("This is a scalene triangle. ", triangle.triangleType());
	}//end test
	
	@Test
	public void test9(){
		Triangle triangle = new Triangle("6","4","5");
		assertEquals("This is a scalene triangle. ", triangle.triangleType());
	}//end test
	
	@Test
	public void test10(){
		Triangle triangle = new Triangle("6","5","4");
		assertEquals("This is a scalene triangle. ", triangle.triangleType());
	}//end test	
	


	/** -------------------------------------------------------------------------------------------------------------------------
	 * test11 tests the default constructor of Triangle, which calls the three-parameter constructor of Triangle with sides of
	 * length 0.
	 * test12 tests the three-parameter constructor of Triangle for sides of length 0.
	 * A line segment with three subsegments of length s1 = s2 = s3 = 0 cannot be a triangle because the conditions s1 + s2 > s3,
	 * s1 + s3 > s2, and s2 + s3 > s1 are not satisfied.
	 ------------------------------------------------------------------------------------------------------------------------- */

	@Test
	public void test11(){
		Triangle triangle = new Triangle();
		assertEquals("Not a valid triangle!\n",triangle.triangleType());
	}//end test

	@Test
	public void test12(){
		Triangle triangle = new Triangle("0","0","0");
		assertEquals("Not a valid triangle!\n",triangle.triangleType());
	}//end test
	

	/** -----------------------------------------------------------------------------------------------------------------------------
	 * The following test cases relate to one or more arguments being negative.
	 * The test cases combine two error messages into one error message. Is this behavior desired?
	 *
	 * Consider integers n1, n2, and n3.
	 * 
	 * Let n1 = n2 = n3.
	 * Let n1 < 0. Then
	 * n1 = n2 = n3 < 0.
	 * n1 + n2 <= n3.
	 * n1 + n3 <= n2.
	 * n2 + n3 <= n1.
	 * 
	 * Thus, for three equal negative integers, it is never the case that all inequality conditions for triangles hold.
	 * 
	 * Let n1 = n2.
	 * Let n1 < 0, n3 < 0. Then
	 * n1 = n2 < 0.
	 * n1 + n3 <= n2.
	 * 
	 * Let n1 = n2.
	 * Let n1 < 0, n3 > 0. Then
	 * n1 = n2 < 0.
	 * n1 + n2 <= n3.
	 * 
	 * Let n1 = n2.
	 * Let n1 > 0, n3 < 0. Then
	 * n1 = n2 > 0.
	 * n1 + n3 <= n2.
	 * 
	 * Thus, for three integers where two are equal and at least one is negative, it is never the case that all inequality conditions
	 * for triangles hold.
	 * 
	 * Let n1 < 0, n2 = 0, n3 < 0. Then
	 * n1 + n3 <= n2.
	 * 
	 * Let n1 < 0, n2 = 0, n3 = 0. Then
	 * n1 + n2 <= n3.
	 * 
	 * Let n1 < 0, n2 = 0, n3 > 0. Then
	 * n1 + n2 <= n3.
	 * 
	 * Thus, for three integers where one is negative and one is zero, it is never the case that all inequality conditions for
	 * triangles hold.
	 * 
	 * Let n1 < 0, n2 < 0, n3 > 0. Then
	 * n1 + n2 <= n3.
	 * 
	 * Let n1 < 0, n2 > 0, n3 > 0. Then
	 * n1 + n2 ? n3.
	 * n1 + n3 ? n2.
	 * n2 + n3 ? n1.
	 * Based on the below simulation, and similar simulations with intervals (-10, -1) and (1, 10) and (-100, -1) and (1, 100),
	 * it is unlikely that there is a case where all threee inequality conditions hold.
     *   RandomDataGenerator randomDataGenerator = new RandomDataGenerator();
	 *   int i, j, k;
     *   while (true) {
     *   	i = randomDataGenerator.nextInt(Integer.MIN_VALUE, -1);
     *   	j = randomDataGenerator.nextInt(1, Integer.MAX_VALUE);
     *   	k = randomDataGenerator.nextInt(1, Integer.MAX_VALUE);
     *   	if (j > Integer.MAX_VALUE - k) { continue; }
	 *		if ((i + j > k) && (i + k > j) && (j + k > i)) {
	 *			System.out.println(i + ", " + j + ", " + k);
	 *			break;
	 *		}
     *   }
	 *
	 * Let n1 < 0, n2 < 0, n3 < 0. Then
	 * n1 + n2 ? n3.
	 * n1 + n3 ? n2.
	 * n2 + n3 ? n1.
	 * Based on the above simulations, it is unlikely that there is a case where all three inequality conditions hold.
	 ----------------------------------------------------------------------------------------------------------------------- */

	/*@Test
	public void verifyThatProvidingAnErrorMessageWithTwoSubmessagesIsDesiredForNegativeArguments() {
		fail("Verify that providing an error message with two submessages is desired.");
	}*/

	// Per the above, this error message works for three integers where one is negative and three are equal.
	@Test
	public void test13(){
		Triangle triangle = new Triangle("-1","-1","-1");
		assertEquals("At least one side is negative!\nNot a valid triangle!\n",triangle.triangleType());
	}//end test
	
	// Per the above, this error message works for three integers where one is negative and two are equal.
	@Test
	public void test14(){
		Triangle triangle = new Triangle("-1","3","3");
		assertEquals("At least one side is negative!\nNot a valid triangle!\n",triangle.triangleType());
	}//end test
	
	// Per the above, this error message works for three integers where one is negative and one is zero.
	@Test
	public void test15(){
		Triangle triangle = new Triangle("-1","0","3");
		assertEquals("At least one side is negative!\nNot a valid triangle!\n",triangle.triangleType());
	}//end test
	
	// Per the above, this error message works for three integers where two are negative.
	@Test
	public void test16(){
		Triangle triangle = new Triangle("-2","-1","3");
		assertEquals("At least one side is negative!\nNot a valid triangle!\n",triangle.triangleType());
	}//end test	
	
	// Per the above, this error message works for three integers where one is negative.
	@Test
	public void test17(){
		Triangle triangle = new Triangle("-1","1","2");
		assertEquals("At least one side is negative!\nNot a valid triangle!\n",triangle.triangleType());
	}//end test
	
	// Per the above, this error message works for three integers where two are negative.
	@Test
	public void test18(){
		Triangle triangle = new Triangle("-2","-1","3");
		assertEquals("At least one side is negative!\nNot a valid triangle!\n",triangle.triangleType());
	}//end test	
	
	// Per the above, this error message works for three negative integers.
	@Test
	public void test19(){
		Triangle triangle = new Triangle("-3","-2","-1");
		assertEquals("At least one side is negative!\nNot a valid triangle!\n",triangle.triangleType());	
	}//end test
	

	/** ------------------------------------------------------------------------------------------
	 * The following test cases relate to one argument being the empty string.
	 * The test cases combine two error messages into one error message. Is this behavior desired?
	 ------------------------------------------------------------------------------------------ */

	/*@Test
	public void verifyThatProvidingAnErrorMessageWithTwoSubmessagesIsDesiredWhenForEmptyStrings() {
		fail("Verify that providing an error message with two submessages is desired.");
	}*/

	@Test
	public void test20(){
		Triangle triangle = new Triangle("","2","3");
		assertEquals("The side 1 is not an integer number.\n\nNot a valid triangle!\n",triangle.triangleType());	
	}//end test

	@Test
	public void test21(){
		Triangle triangle = new Triangle("1","","3");
		assertEquals("The side 2 is not an integer number.\n\nNot a valid triangle!\n",triangle.triangleType());
	}//end test

	@Test
	public void test22(){
		Triangle triangle = new Triangle("1","2","");
		assertEquals("The side 3 is not an integer number.\n\nNot a valid triangle!\n",triangle.triangleType());
	}//end test


	/** --------------------------------------------------------------------------------------------
	 * The following test cases relate to two arguments being the empty string.
	 * The test cases combine three error messages into one error message. Is this behavior desired?
	 -------------------------------------------------------------------------------------------- */

	@Test
	public void test23(){
		Triangle triangle = new Triangle("","","3");
		assertEquals(
			"The side 1 is not an integer number.\n\n" +
			"The side 2 is not an integer number.\n\n" +
			"Not a valid triangle!\n",
			triangle.triangleType()
		);
	}//end test

	@Test
	public void test24(){
		Triangle triangle = new Triangle("","2","");
		assertEquals(
			"The side 1 is not an integer number.\n\n" +
			"The side 3 is not an integer number.\n\n" +
			"Not a valid triangle!\n",
			triangle.triangleType()
		);
	}//end test

	@Test
	public void test25(){
		Triangle triangle = new Triangle("1","","");
		assertEquals(
			"The side 2 is not an integer number.\n\n" +
			"The side 3 is not an integer number.\n\n" +
			"Not a valid triangle!\n",
			triangle.triangleType()
		);
	}//end test

	@Test
	public void test26(){
		Triangle triangle = new Triangle("","","");
		assertEquals(
			"The side 1 is not an integer number.\n\n" +
			"The side 2 is not an integer number.\n\n" +
			"The side 3 is not an integer number.\n\n" +
			"Not a valid triangle!\n",
			triangle.triangleType()
		);	
	}//end test


	/** --------------------------------------------------------------------------------
	 * The following test relates to an argument having a character that is not a digit.
	 -------------------------------------------------------------------------------- */

	@Test
	public void test27(){
		Triangle triangle = new Triangle("00.1","2","3");
		assertEquals("The side 1 is not an integer number.\n\nNot a valid triangle!\n",triangle.triangleType());
	}//end test


	/** ------------------------------------------------------
	 * The following test relates to a triangle being too big.
	 ------------------------------------------------------ */

	@Test
	public void test28(){
		Triangle triangle = new Triangle("300","400","500");
		assertEquals("This triangle is too big.\n",triangle.triangleType());	
	}//end test

}
