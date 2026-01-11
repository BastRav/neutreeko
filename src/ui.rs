use std::rc::Rc;

use yew::prelude::*;
use yew::{html, Component, Context, Html};

use crate::logic::{Board, Direction, Pawn, Position};

pub enum Msg {
    PawnClick(usize),
    Left,
    Right,
    Up,
    Down,
    UpLeft,
    UpRight,
    DownLeft,
    DownRight,
}

pub struct App {
    board: Board,
    state: Rc<AppState>,
    direction: Direction
}

#[derive(Clone, PartialEq)]
pub struct AppState {
    pawn_clicked: Callback<usize>,
    last_clicked: Option<usize>
}

impl Component for App {
    type Message = Msg;
    type Properties = ();

    fn create(ctx: &Context<Self>) -> Self {
        let pawn_clicked = ctx.link().callback(Msg::PawnClick);
        let state = Rc::new(AppState {
            pawn_clicked,
            last_clicked: None
        });

        let board = Board::default_new();
        Self {
            board: board,
            state: state,
            direction: Direction::Up
        }
    }

    fn update(&mut self, _ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::PawnClick(pawn_index) => {
                let shared_state = Rc::make_mut(&mut self.state);
                shared_state.last_clicked = Some(pawn_index);

                let mut new_board = self.board.clone();
                match new_board.move_pawn_until_blocked(pawn_index, &self.direction) {
                    true => {
                        self.board = new_board;
                    },
                    false => ()
                };
            }
            Msg::Up => {
                self.direction = Direction::Up;
            }
            Msg::Down => {
                self.direction = Direction::Down;
            }
            Msg::Left => {
                self.direction = Direction::Left;
            }
            Msg::Right => {
                self.direction = Direction::Right;
            }
            Msg::UpLeft => {
                self.direction = Direction::UpLeft;
            }
            Msg::UpRight => {
                self.direction = Direction::UpRight;
            }
            Msg::DownLeft => {
                self.direction = Direction::DownLeft;
            }
            Msg::DownRight => {
                self.direction = Direction::DownRight;
            }
        }
        true
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let app_state = self.state.clone();
        let mut next_player_text;
        match self.board.next_player {
            crate::logic::Color::Green => next_player_text = "Green".to_string() + "'s turn",
            crate::logic::Color::Yellow => next_player_text = "Yellow".to_string() + "'s turn",
        }
        match self.board.winner(){
            Some(color) => {
                match color {
                    crate::logic::Color::Green => next_player_text = "Green".to_string() + " wins!",
                    crate::logic::Color::Yellow => next_player_text = "Yellow".to_string() + " wins!",
                }
            }
            None => ()
        }
        html! {
            <ContextProvider<Rc<AppState>> context={app_state}>
                <div>
                    <button onclick={ctx.link().callback(|_| Msg::Up)}>{ "Up" }</button>
                    <button onclick={ctx.link().callback(|_| Msg::Down)}>{ "Down" }</button>
                    <button onclick={ctx.link().callback(|_| Msg::Left)}>{ "Left" }</button>
                    <button onclick={ctx.link().callback(|_| Msg::Right)}>{ "Right" }</button>
                    <button onclick={ctx.link().callback(|_| Msg::UpLeft)}>{ "UpLeft" }</button>
                    <button onclick={ctx.link().callback(|_| Msg::UpRight)}>{ "UpRight" }</button>
                    <button onclick={ctx.link().callback(|_| Msg::DownLeft)}>{ "DownLeft" }</button>
                    <button onclick={ctx.link().callback(|_| Msg::DownRight)}>{ "DownRight" }</button>
                    <p>{ format!("Next move direction {:?}", self.direction) }</p>
                </div>
                <h2>{ next_player_text }</h2>
                <BoardView board={self.board.clone()} />
            </ContextProvider<Rc<AppState>>>
        }
    }

}

pub struct PawnView{
    state: Rc<AppState>,
    _listener: ContextHandle<Rc<AppState>>
}

#[derive(Clone, Properties, PartialEq)]
pub struct PawnComponent {
    pub pawn: Pawn,
    pub position: Position,
    pub index: usize
}

const SCALING: u32 = 80;
const FROM_TOP: u32 = 220;

const MARGIN: u32 = 5;

pub enum PawnMsg {
    ContextChanged(Rc<AppState>),
}

impl Component for PawnView {
    type Message = PawnMsg;
    type Properties = PawnComponent;
 
    fn create(ctx: &Context<Self>) -> Self {
        let (state, _listener) = ctx
            .link()
            .context::<Rc<AppState>>(ctx.link().callback(PawnMsg::ContextChanged))
            .expect("context to be set");

        Self { state, _listener }
    }

    fn update(&mut self, _ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            PawnMsg::ContextChanged(state) => {
                self.state = state;
                true
            }
        }
    }
 
    fn view(&self, ctx: &Context<Self>) -> Html {
        let my_index = ctx.props().index;
        let onclick = self.state.pawn_clicked.reform(move |_| my_index);
        html! {
            <div
                onclick={onclick}
                style={format!(
                    "width: {}px; height: {}px; background-color: {}; position: absolute; top: {}px; left: {}px; border-radius: 50%; border: 2px solid black; cursor: pointer;",
                    SCALING - MARGIN * 2,
                    SCALING - MARGIN * 2,
                    match ctx.props().pawn.color {
                        crate::logic::Color::Green => "green",
                        crate::logic::Color::Yellow => "yellow",
                    },
                    u32::from(ctx.props().position.row) * SCALING + MARGIN - 1,
                    u32::from(ctx.props().position.column) * SCALING + MARGIN - 1,
                )}
            >
            </div>
        }
    }
}

pub struct BoardView;

#[derive(Clone, Properties, PartialEq)]
pub struct BoardComponent {
    pub board: Board,
}
 
impl Component for BoardView {
    type Message = ();
    type Properties = BoardComponent;
 
    fn create(_ctx: &Context<Self>) -> Self {
        Self {}
    }
 
    fn view(&self, ctx: &Context<Self>) -> Html {
        let mut pawns = Vec::new();
        for (index, pawn) in ctx.props().board.pawns.iter().enumerate() {
            pawns.push(html! {
                <PawnView pawn={pawn.clone()} position={pawn.position.clone()} index={index} />
            });
        }
        html! {
            <div style={format!(
                "position: absolute; top: {}px; width: {}px; height: {}px; background-image: linear-gradient(0deg, #e0e0e0 1px, transparent 1px), linear-gradient(90deg, #e0e0e0 1px, transparent 1px); background-size: {}px {}px; background-position: 0 0; border-top: 1px solid #e0e0e0; border-left: 1px solid #e0e0e0; border-right: 1px solid #e0e0e0; border-bottom: 1px solid #e0e0e0;",
                FROM_TOP,
                SCALING * ctx.props().board.number_of_columns as u32,
                SCALING * ctx.props().board.number_of_rows as u32,
                SCALING, SCALING
            )}>
                {pawns}
            </div>
        }
    }
}
