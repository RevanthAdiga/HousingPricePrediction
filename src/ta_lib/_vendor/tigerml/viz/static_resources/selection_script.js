function simulateClick(item) {
  item.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true}));
  item.dispatchEvent(new MouseEvent('mousedown', {bubbles: true}));
  item.dispatchEvent(new PointerEvent('pointerup', {bubbles: true}));
  item.dispatchEvent(new MouseEvent('mouseup', {bubbles: true}));
  item.dispatchEvent(new MouseEvent('mouseout', {bubbles: true}));
  item.dispatchEvent(new MouseEvent('click', {bubbles: true}));
  item.dispatchEvent(new Event('change', {bubbles: true}));
  return true;
}

function clickOption(select_element, index) {
    var optionToClick = document.querySelector(select_element).children[index]; //choose any of the children
    optionToClick.selected = true;
    simulateClick(optionToClick); // manual click simulation
}

//var bk_links = document.querySelectorAll(".selections a"); // this element contains more than 1 DOMs.
//for(var i =0; i < bk_links.length; i++) {
//    (function(i) {
//        bk_links[i].onclick = function() {
//            clickOption('.selection_input select', i);
//            document.querySelectorAll(".load_selection button")[0].click();
//        };
//    })(i);
//}

var bk_edits = document.querySelectorAll(".selections .icon-edit"); // this element contains more than 1 DOMs.
for(var i =0; i < bk_edits.length; i++) {
    (function(i) {
        bk_edits[i].onclick = function() {
            clickOption('.selection_input select', i);
            document.querySelectorAll(".edit_selection_widget .selection_name input")[0].value =
            document.querySelectorAll(".selections a")[i].innerHTML;
            document.querySelectorAll(".edit_selection_widget")[0].classList.remove('is_hidden');
        };
    })(i);
}

var bk_expands = document.querySelectorAll(".selections .icon-expand"); // this element contains more than 1 DOMs.
for(var i =0; i < bk_expands.length; i++) {
    (function(i) {
        bk_expands[i].onclick = function() {
            clickOption('.selection_input select', i);
            document.querySelectorAll(".expand_selection button")[0].click();
        };
    })(i);
}

el = document.querySelectorAll(".edit_selection_widget .icon-ok")[0]
el.onclick = function() {
    document.querySelectorAll(".edit_selection_widget")[0].classList.add('is_hidden');
    document.querySelectorAll(".edit_selection button")[0].click();
}

el = document.querySelectorAll(".edit_selection_widget .icon-cancel")[0]
el.onclick = function() {
    document.querySelectorAll(".edit_selection_widget")[0].classList.add('is_hidden');
}

overlays = document.querySelectorAll(".overlay")
for(var i =0; i < overlays.length; i++) {
    (function(i) {
        overlays[i].querySelector('.glass').onclick = function() {
            overlays[i].classList.add('is_hidden');
        };
    })(i);
}

//el = document.querySelectorAll(".selections_widget .icon-import button")[0]
//el.onclick = function() {
//    document.querySelectorAll(".selections_widget .import_file input")[0].click();
//}

