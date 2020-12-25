/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.serverutilities;

import java.time.LocalDateTime;

/**
 *
 * @author thoma
 */
public class Controller {
    
    public Controller() {
        
    }
    
    public void process() {
        
        System.out.printf("Hi from process at %s!\n", LocalDateTime.now());
        
    }
    
}
