//----------------------------------------------------------------------
// Balanced.java           by Dale/Joyce/Weems                 Chapter 2
//
// Checks for balanced expressions using standard rules.
//
// Matching pairs of open and close symbols are provided to the
// constructor through two string parameters.
//----------------------------------------------------------------------
package ch02.balanced;

import ch02.stacks.*;

public class Balanced 
{
  protected String openSet;
  protected String closeSet;

  public Balanced(String openSet, String closeSet)
  // Preconditions: No character is contained more than once in the 
  //                combined openSet and closeSet strings.
  //                The size of openSet = the size of closeSet.
  {
    this.openSet = openSet;
    this.closeSet = closeSet;
  }

  public int test(String expression)
  // Returns 0 if expression is balanced.
  // Returns 1 if expression has unbalanced symbols.
  // Returns 2 if expression came to end prematurely.
  {
    char currChar;                   // current expression character being studied
    int  currCharIndex;              // index of current character
    int  lastCharIndex;              // index of last character in the expression

    int openIndex;                   // index of current character in openSet
    int closeIndex;                  // index of current character in closeSet
    
    boolean stillBalanced = true;    // true as long as expression is balanced

    StackInterface<Integer> stack;   // holds unmatched open symbols
    stack = new ArrayBoundedStack<Integer>(expression.length());

    currCharIndex = 0;
    lastCharIndex = expression.length() - 1;

    while (stillBalanced && (currCharIndex <= lastCharIndex))
    // while expression still balanced and not at end of expression
    {
      currChar = expression.charAt(currCharIndex);
      openIndex = openSet.indexOf(currChar);

      if(openIndex != -1)   // if the current character is in the openSet
      {
        // Push the index onto the stack.
        stack.push(openIndex);
      }
        else
        {
          closeIndex = closeSet.indexOf(currChar);
          if(closeIndex != -1)     // if the current character is in the closeSet 
          {
            try                    // try to pop an index off the stack
            {
              openIndex = stack.top();
              stack.pop();     
              if (openIndex != closeIndex)   // if popped index doesn't match
                stillBalanced = false;         // then expression is not balanced
            }
            catch(StackUnderflowException e) // if stack was empty
            {
              stillBalanced = false;           // then expression is not balanced
            }
          }
        }
        currCharIndex++;             // set up processing of next character
      }

      if (!stillBalanced)
        return 1;             // unbalanced symbols
      else
      if (!stack.isEmpty())
        return 2;             // premature end of expression
      else
        return 0;             // expression is balanced
  }
}
