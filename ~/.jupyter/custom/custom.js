// Enable line numbers in all code cells
define([
    'base/js/namespace',
    'base/js/events'
], function(Jupyter, events) {
    "use strict";
    
    // Function to enable line numbers
    var enable_line_numbers = function() {
        Jupyter.notebook.get_cells().forEach(function(cell) {
            if (cell.cell_type === 'code') {
                cell.code_mirror.setOption('lineNumbers', true);
            }
        });
    };
    
    // Enable line numbers when notebook is loaded
    events.on('notebook_loaded.Notebook', enable_line_numbers);
    
    // Enable line numbers for new cells
    events.on('create.Cell', function(event, data) {
        if (data.cell.cell_type === 'code') {
            data.cell.code_mirror.setOption('lineNumbers', true);
        }
    });
    
    // If notebook is already loaded, enable now
    if (Jupyter.notebook && Jupyter.notebook._fully_loaded) {
        enable_line_numbers();
    }
    
    console.log("Custom cell numbers and line numbers enabled!");


    {
        "editor.lineNumbers": "on",
        "notebook.lineNumbers": "on",
        "notebook.showCellStatusBar": true,
        "notebook.cellToolbarLocation": "left",
        "notebook.outline.showCodeCells": true,
        "notebook.breadcrumbs.showNavigationPath": true,
        "notebook.cellFocusIndicator": "gutter"
    }

}); 