from student import Student
import unittest

class EnrollInCourseTest(unittest.TestCase): 
    
    def test_incremented_correctly(self):
        """
        Test if enroll_in_course() method successfully increments the
        num_courses attribute of the Student object
        """

        # Create student instance and add some courses
        student1 = Student('Katherine', ['DS 5100'])
        student1.enroll_in_course("CS 5050")
        student1.enroll_in_course("CS 5777")
        print(student1.courses)
        print(student1.num_courses)
        
        # Test
        expected = 3
        # Note that unittest.TestCase brings in the assertEqual() method
        self.assertEqual(student1.num_courses, expected)

    # def test_test(self):
    #     self.assertTrue(False)
        
if __name__ == '__main__':
    unittest.main(verbosity=2)