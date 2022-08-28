//---------------------------------------------------------------------
// PalindromeGUI.java       by Dale/Joyce/Weems               Chapter 4
//
// Checks for strings that are palidromes.
// Input consists of a sequence of strings.
// Output consists of whether the input string is a palindrome.
// Interacts with the user through a graphical user interface.
//----------------------------------------------------------------------
package ch04.apps;

import java.awt.*;            
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import java.io.*;
import ch04.palindromes.Palindrome;

public class PalindromeGUI 
{
  // text field
  private static JTextField expressionText;  // text field for postfix expression

  // status Label
  private static JLabel statusLabel;         // label for status/result info

  // Define a button listener.
  private static class ActionHandler implements ActionListener 
  {
    public void actionPerformed(ActionEvent event)
    // listener for the button events
    {
      if (event.getActionCommand().equals("Evaluate"))
      { // Handles Evaluate event.
        int result = 0;
        if (Palindrome.test(expressionText.getText()))
          statusLabel.setText("is a palindrome.\n");
        else
          statusLabel.setText("is NOT a palindrome.\n");  
      }
      else
      if (event.getActionCommand().equals("Clear"))
      { // Handles Clear event.
        statusLabel.setText("cleared");
        expressionText.setText("");
      }
    }
  }

  public static void main(String[] args) 
  {
    // Declare/instantiate/initialize display frame.
    JFrame displayFrame = new JFrame();
    displayFrame.setTitle("Palindrome Evaluator Program");
    displayFrame.setSize(400,100);
    displayFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

    // text box for expression
    expressionText = new JTextField("potential palindrome goes here", 60);

    // Status/Result label
    statusLabel = new JLabel("status", JLabel.CENTER);
    statusLabel.setBorder(new LineBorder(Color.red,3));

    // Evaluate and clear buttons    
    JButton evaluate   = new JButton("Evaluate");         
    JButton clear       = new JButton("Clear");        

    // Button event listener
    ActionHandler action = new ActionHandler();
 
    // Register button listeners.
    evaluate.addActionListener(action);
    clear.addActionListener(action);

    // Instantiate content pane and information panels.
    Container contentPane = displayFrame.getContentPane();
    JPanel expressionPanel = new JPanel();
    JPanel buttonPanel = new JPanel();
  
    // Initialize expression panel.
    expressionPanel.setLayout(new GridLayout(2,1));
    expressionPanel.add(expressionText);
    expressionPanel.add(statusLabel);

    // Initialize button panel.
    buttonPanel.setLayout(new GridLayout(1,2));
    buttonPanel.add(evaluate);
    buttonPanel.add(clear);

     // Set up and show the frame.
    contentPane.add(expressionPanel, "North");
    contentPane.add(buttonPanel, "Center");
 
    displayFrame.pack();
    displayFrame.setVisible(true);
  }
}