//----------------------------------------------------------------------------
// LargeIntGUI.java             by Dale/Joyce/Weems                  Chapter 6
//
// Evaluates addition and subtraction of large integers
//----------------------------------------------------------------------------
package ch06.apps;

import ch06.largeInts.*;
import java.awt.*;            
import java.awt.event.*;
import javax.swing.*;
import java.io.*;

public class LargeIntGUI
{
  // large integers
  private static LargeInt first;
  private static LargeInt second;
  private static LargeInt result;

  // text field
  private static JTextField operandAText;  // text field for operand A
  private static JTextField operandBText;  // text field for operand B

  // status Label
  private static JLabel statusLabel;         // Label for status info
  private static JLabel resultLabel;         // Label for status info

  // Radio Buttons and Group for choosing operation
  private static JRadioButton plusButton;
  private static JRadioButton minusButton;
  private static ButtonGroup operationGroup;

  // Define a button listener
  private static class ActionHandler implements ActionListener 
  {
    public void actionPerformed(ActionEvent event)
    // Listener for the button events
    {
      if (event.getActionCommand().equals("Calculate"))
      { // Handles Calculate event

        first = new LargeInt(operandAText.getText());
        second =  new LargeInt(operandBText.getText());

        result = LargeInt.add(first, second);

        String choice = operationGroup.getSelection().getActionCommand();
        if (choice == "plus")
        {
          statusLabel.setText("The sum of the first and second operands is "); 
          result = LargeInt.add(first, second);
        }
        else
        {
          statusLabel.setText("The difference of the first and second operands is "); 
          result = LargeInt.subtract(first, second);
        }
        resultLabel.setText(result.toString());
      }
      else
      if (event.getActionCommand().equals("Clear"))
      { // Handles Clear event
        statusLabel.setText("cleared");
        resultLabel.setText("cleared");
        operandAText.setText("");
        operandBText.setText("");
      }
    }
  }

  public static void main(String args[]) throws IOException
  {
    // Declare/instantiate/initialize display frame
    JFrame displayFrame = new JFrame();
    displayFrame.setTitle("Large Integer Calculator Program");
    displayFrame.setSize(600,250);
//    displayFrame.setDefaultCloseOperation(3); 
    displayFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); 

    // text box for operands
    operandAText = new JTextField( 60);
    operandBText = new JTextField( 60);

    // Radio Buttons for choosing operation
    JRadioButton plusButton = new JRadioButton("   First + Second   ");
    plusButton.setSelected(true);
    JRadioButton minusButton = new JRadioButton("   First - Second   ");
    plusButton.setActionCommand("plus");
    minusButton.setActionCommand("minus");
    operationGroup = new ButtonGroup();
    operationGroup.add(plusButton);
    operationGroup.add(minusButton);

    // Status/Result labels
    statusLabel = new JLabel("Message will go here", JLabel.CENTER);
    resultLabel = new JLabel("Result will go here", JLabel.CENTER);

    // Various Labels 
    JLabel operandALabel   = new JLabel("First Operand", JLabel.LEFT);           
    JLabel operandBLabel   = new JLabel("Second Operand", JLabel.LEFT);           
    JLabel operatorsLabel   = new JLabel("Choose an Operation:", JLabel.LEFT);       
    JLabel blankLabel1 = new JLabel("");
    JLabel blankLabel2 = new JLabel("");
    JLabel blankLabel3 = new JLabel("");
    
    // Calculate and clear buttons    
    JButton calculate   = new JButton("Calculate");         
    JButton clear       = new JButton("Clear");        

    // Button event listener
    ActionHandler action = new ActionHandler();
 
    // Register button listeners
    calculate.addActionListener(action);
    clear.addActionListener(action);

    // Instantiate content pane and information panels
    Container contentPane = displayFrame.getContentPane();
    JPanel setupPanel = new JPanel();
    JPanel operatorPanel = new JPanel();
    JPanel resultPanel = new JPanel();
    JPanel buttonPanel = new JPanel();
 
    // Initialize setup panel
    setupPanel.setLayout(new GridLayout(6,1));
    setupPanel.add(operandALabel);
    setupPanel.add(operandAText);
    setupPanel.add(operandBLabel);
    setupPanel.add(operandBText);
    setupPanel.add(operatorsLabel);
    operatorPanel.setLayout(new GridLayout(1,5));
    operatorPanel.add(blankLabel1);
    operatorPanel.add(plusButton);
    operatorPanel.add(blankLabel2);
    operatorPanel.add(minusButton);
    operatorPanel.add(blankLabel3);
    setupPanel.add(operatorPanel);

    // Initialize result panel
    resultPanel.setLayout(new GridLayout(2,1));
    resultPanel.add(statusLabel);
    resultPanel.add(resultLabel);

    // Initialize button panel
    buttonPanel.setLayout(new GridLayout(1,2));
    buttonPanel.add(calculate);
    buttonPanel.add(clear);

    // set up and show the frame
    contentPane.add(setupPanel, "North");
    contentPane.add(resultPanel, "Center");
    contentPane.add(buttonPanel, "South");

    displayFrame.pack();
    displayFrame.setVisible(true);
  }
}