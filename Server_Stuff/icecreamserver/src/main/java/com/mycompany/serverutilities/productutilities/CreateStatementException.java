/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.serverutilities.productutilities;

/**
 *
 * @author thoma
 */
public class CreateStatementException extends Exception {
    public CreateStatementException(String errorMessageToUse) {
        super(errorMessageToUse);
    }
}
