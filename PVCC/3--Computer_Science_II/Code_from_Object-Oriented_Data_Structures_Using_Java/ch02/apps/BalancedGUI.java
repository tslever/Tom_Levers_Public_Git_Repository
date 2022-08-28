//----------------------------------------------------------------------
// BalancedGUI.java         by Dale/Joyce/Weems                Chapter 2
//
// Checks for balanced grouping symbols.
// Special symbol types are (), [], and {}.
// Uses a graphical user interface.
//----------------------------------------------------------------------
package ch02.apps;

import java.awt.*;            
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import java.io.*;
import ch02.balanced.*;

public class BalancedGUI
{
  // text field
  private static JTextField expressionText;  // text field for expression

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
      
        // Instantiate new Balanced class with grouping symbols.
        Balanced bal = new Balanced("([{", ")]}");
    
        int result; // 0 = balanced, 1 = unbalanced, 2 = premature end

        result = bal.test(expressionText.getText());
        if (result == 1)
           statusLabel.setText("Result = Unbalanced \n");
        else
        if (result == 2)
           statusLabel.setText("Result = Premature end of expression \n");
        else
           statusLabel.setText("Result = Balanced \n"); 
      }
      else
      if (event.getActionCommand().equals("Clear"))
      { // Handles Clear event.
        statusLabel.setText("cleared");
        expressionText.setText("");
      }
    }
  }

  public static void main(String args[]) throws IOException
  {
    // Declare/instantiate/initialize display frame.
    JFrame displayFrame = new JFrame();
    displayFrame.setTitle("Balanced Expression Evaluator Program");
    displayFrame.setSize(400,100);
    displayFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

    // text box for expression
    expressionText = new JTextField("expression goes here", 60);

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
