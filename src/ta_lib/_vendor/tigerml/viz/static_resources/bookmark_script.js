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

var bk_links = document.querySelectorAll(".bookmarks a"); // this element contains more than 1 DOMs.
for(var i =0; i < bk_links.length; i++) {
    (function(i) {
        bk_links[i].onclick = function() {
            clickOption('.bookmark_input select', i);
            document.querySelectorAll(".load_bookmark button")[0].click();
        };
    })(i);
}

var bk_edits = document.querySelectorAll(".bookmarks .icon-edit"); // this element contains more than 1 DOMs.
for(var i =0; i < bk_edits.length; i++) {
    (function(i) {
        bk_edits[i].onclick = function() {
            clickOption('.bookmark_input select', i);
            document.querySelectorAll(".edit_bookmark_widget .bookmark_name input")[0].value = bk_links[i].innerHTML;
            document.querySelectorAll(".edit_bookmark_widget")[0].classList.remove('is_hidden');
        };
    })(i);
}

var bk_deletes = document.querySelectorAll(".bookmarks .icon-delete"); // this element contains more than 1 DOMs.
for(var i =0; i < bk_deletes.length; i++) {
    (function(i) {
        bk_deletes[i].onclick = function() {
            clickOption('.bookmark_input select', i);
            document.querySelectorAll(".delete_bookmark button")[0].click();
        };
    })(i);
}

el = document.querySelectorAll(".edit_bookmark_widget .icon-ok")[0]
el.onclick = function() {
    document.querySelectorAll(".edit_bookmark_widget")[0].classList.add('is_hidden');
    document.querySelectorAll(".edit_bookmark button")[0].click();
}

el = document.querySelectorAll(".edit_bookmark_widget .icon-cancel")[0]
el.onclick = function() {
    document.querySelectorAll(".edit_bookmark_widget")[0].classList.add('is_hidden');
}

el = document.querySelectorAll(".bookmarks_widget .icon-import button")[0]
el.onclick = function() {
    document.querySelectorAll(".bookmarks_widget .import_file input")[0].click();
}

