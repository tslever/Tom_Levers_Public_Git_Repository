class Student:
    
    # constructor
    def __init__(self, name, courses=None):
        self.name = name # string type
        self.courses = [] if courses is None else courses # list of strings
        self.num_courses = len(self.courses)
        
    # enroll in a course
    def enrollInCourse(self, cname): 
        self.courses.append(cname)
        self.num_courses += 1 # increment the number of courses

    def unenroll_in_course(self, cname):
        if cname in self.courses:
            self.courses.remove(cname)
        else:
            raise Exception(self.name + ' is not enrolled in ' + cname + '!')